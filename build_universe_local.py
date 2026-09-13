import json

print("[*] Bypassing broken Angel One Scrip Master server...")
print("[*] Generating local fno_universe.json...")

# Top 25 High-Volume F&O Anchors (Direct Mapping)
universe = [
    {"symbol": "RELIANCE-EQ", "token": "2885"},
    {"symbol": "HDFCBANK-EQ", "token": "1333"},
    {"symbol": "ICICIBANK-EQ", "token": "4963"},
    {"symbol": "INFY-EQ", "token": "1594"},
    {"symbol": "TCS-EQ", "token": "11536"},
    {"symbol": "ITC-EQ", "token": "1660"},
    {"symbol": "SBIN-EQ", "token": "4329"},
    {"symbol": "BHARTIARTL-EQ", "token": "10604"},
    {"symbol": "BAJFINANCE-EQ", "token": "317"},
    {"symbol": "L&T-EQ", "token": "11483"},
    {"symbol": "KOTAKBANK-EQ", "token": "1922"},
    {"symbol": "AXISBANK-EQ", "token": "5900"},
    {"symbol": "HUL-EQ", "token": "1394"},
    {"symbol": "MARUTI-EQ", "token": "10999"},
    {"symbol": "SUNPHARMA-EQ", "token": "3351"},
    {"symbol": "ULTRACEMCO-EQ", "token": "11532"},
    {"symbol": "TITAN-EQ", "token": "3506"},
    {"symbol": "M&M-EQ", "token": "2031"},
    {"symbol": "ASIANPAINT-EQ", "token": "236"},
    {"symbol": "NTPC-EQ", "token": "11630"},
    {"symbol": "TATASTEEL-EQ", "token": "3499"},
    {"symbol": "TATAMOTORS-EQ", "token": "3456"},
    {"symbol": "POWERGRID-EQ", "token": "14977"},
    {"symbol": "WIPRO-EQ", "token": "3787"},
    {"symbol": "INDUSTOWER-EQ", "token": "29135"}
]

with open("fno_universe.json", "w") as f:
    json.dump(universe, f, indent=2)

print(f"[✓] Successfully built local F&O universe with {len(universe)} high-liquidity anchors.")
print("[✓] You may now run the screener.")
