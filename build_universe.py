import sys
import json
import time
import requests

def build():
    url = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"
    print("[*] Downloading Angel One Scrip Master (~60MB)...")
    
    for attempt in range(1, 4):
        try:
            print(f"[*] Connection attempt {attempt}/3...")
            # 15 seconds to connect, 120 seconds to read data
            response = requests.get(url, stream=True, timeout=(15, 120))
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024 * 1024  # 1MB chunks
            downloaded = 0
            data = bytearray()
            
            for data_chunk in response.iter_content(block_size):
                if data_chunk:
                    data.extend(data_chunk)
                    downloaded += len(data_chunk)
                    if total_size:
                        done = int(50 * downloaded / total_size)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {downloaded/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB")
                        sys.stdout.flush()
            
            print("\n[*] Download complete. Parsing 90,000+ tokens...")
            parsed = json.loads(data.decode('utf-8'))
            
            print("[*] Filtering for active NSE F&O equities...")
            fno_names = {item["name"] for item in parsed if item.get("exch_seg") == "NFO" and item.get("instrumenttype") == "FUTSTK"}
            universe = [
                {"symbol": item["name"], "token": item["token"]} 
                for item in parsed 
                if item.get("exch_seg") == "NSE" and item.get("name") in fno_names and item.get("symbol", "").endswith("-EQ")
            ]
            
            with open("fno_universe.json", "w") as f:
                json.dump(universe, f, indent=2)
                
            print(f"[✓] Successfully built fno_universe.json with {len(universe)} tokens.")
            return
            
        except Exception as e:
            print(f"\n[!] Attempt {attempt} failed: {e}")
            if attempt < 3:
                print("[*] Retrying in 3 seconds...")
                time.sleep(3)
    
    print("[!] Failed to download universe after 3 attempts.")

if __name__ == "__main__":
    build()
