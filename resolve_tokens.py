import json
import urllib.request
import pandas as pd

SCRIP_MASTER_URL = "https://margincalculator.angelbroking.com/OpenAPI_File/files/OpenAPIScripMaster.json"

WATCHLIST = [
    "RAMBHAJO",
    "IIFL",
    "FMGOETZE",
    "AGIIL",
    "RSL",
    "TEJASNET"
]

def fetch_instrument_tokens():
    print("[*] Fetching latest Scrip Master from Angel One CDN...")
    req = urllib.request.Request(SCRIP_MASTER_URL, headers={"User-Agent": "Mozilla/5.0"})
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    
    df = pd.DataFrame(data)
    print(f"[✓] Loaded {len(df):,} total instrument definitions.")

    # Filter for NSE Equity segment only
    nse_eq = df[(df["exch_seg"] == "NSE") & (df["symbol"].str.endswith("-EQ"))].copy()
    nse_eq["clean_symbol"] = nse_eq["symbol"].apply(lambda s: s.replace("-EQ", ""))

    resolved_tokens = []
    print("\n=== RESOLVED WATCHLIST TOKENS ===")
    for target in WATCHLIST:
        match = nse_eq[nse_eq["clean_symbol"] == target]
        if not match.empty:
            row = match.iloc[0]
            token = str(row["token"])
            sym = row["clean_symbol"]
            resolved_tokens.append({"symbol": sym, "token": token})
            print(f"  • {sym:<12} -> Token: {token:<8} | TradingSymbol: {row['symbol']}")
        else:
            print(f"  [X] {target:<10} -> NOT FOUND IN NSE-EQ SEGMENT")

    # Output ready-to-paste Python structure
    print("\n[*] Python list configuration for run_live_session.py:")
    print("SUBSCRIPTION_TOKENS = " + json.dumps(resolved_tokens, indent=4))

    with open("resolved_tokens.json", "w") as f:
        json.dump(resolved_tokens, f, indent=4)
    print("\n[✓] Saved to resolved_tokens.json")

if __name__ == "__main__":
    fetch_instrument_tokens()
