import glob
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

def get_verified_parquet_files(pattern: str) -> list[str]:
    raw_files = glob.glob(pattern)
    verified = []
    for f in raw_files:
        try:
            pq.ParquetFile(f)
            verified.append(f)
        except Exception:
            continue
    return verified

def evaluate_single_file(model, file_path: str):
    try:
        df = pd.read_parquet(file_path)
        df = df[(df['bid_price_0'] > 0) & (df['ask_price_0'] > 0)].copy()
        df = df[df['ask_price_0'] > df['bid_price_0']].copy()
        if len(df) < 200:
            return None

        # Features
        df['spread'] = df['ask_price_0'] - df['bid_price_0']
        df['mid_price'] = (df['ask_price_0'] + df['bid_price_0']) / 2.0
        df['spread_bps'] = (df['spread'] / df['mid_price']) * 10000.0
        df['obi_l1'] = (df['bid_qty_0'] - df['ask_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
        
        bid_vols = df[['bid_qty_0', 'bid_qty_1', 'bid_qty_2', 'bid_qty_3', 'bid_qty_4']].sum(axis=1)
        ask_vols = df[['ask_qty_0', 'ask_qty_1', 'ask_qty_2', 'ask_qty_3', 'ask_qty_4']].sum(axis=1)
        df['obi_deep'] = (bid_vols - ask_vols) / (bid_vols + ask_vols + 1e-6)
        
        micro = (df['bid_price_0'] * df['ask_qty_0'] + df['ask_price_0'] * df['bid_qty_0']) / (df['bid_qty_0'] + df['ask_qty_0'] + 1e-6)
        df['micro_delta_bps'] = ((micro - df['mid_price']) / df['mid_price']) * 10000.0
        cleaned = df.dropna().reset_index(drop=True)

        position = 0
        entry_price = 0.0
        realized_pnl = 0.0
        trades = 0
        wins = 0
        ROUNDTRIP_FEE_BPS = 3.0

        for i in range(len(cleaned) - 1):
            row = cleaned.iloc[i]
            mid = float(row['mid_price'])
            bid = float(row['bid_price_0'])
            ask = float(row['ask_price_0'])

            unrealized_bps = 0.0
            if position == 1 and entry_price > 0:
                unrealized_bps = ((mid - entry_price) / max(entry_price, 1e-6)) * 10000.0
                # Risk Daemon Ejection
                if unrealized_bps <= -20.0:
                    net_diff = (bid - entry_price) - (entry_price * (ROUNDTRIP_FEE_BPS / 10000.0))
                    realized_pnl += net_diff
                    trades += 1
                    position = 0
                    entry_price = 0.0
                    continue

            obs = np.array([
                row['spread_bps'],
                row['micro_delta_bps'],
                row['obi_l1'],
                row['obi_deep'],
                float(position),
                unrealized_bps
            ], dtype=np.float32)

            target_pos, _ = model.predict(obs, deterministic=True)

            # Buy transition
            if position == 0 and target_pos == 1 and ask > 0:
                position = 1
                entry_price = ask

            # Sell transition
            elif position == 1 and target_pos == 0 and entry_price > 0:
                net_diff = (bid - entry_price) - (entry_price * (ROUNDTRIP_FEE_BPS / 10000.0))
                realized_pnl += net_diff
                trades += 1
                if net_diff > 0:
                    wins += 1
                position = 0
                entry_price = 0.0

        # Close any lingering trade at session end
        if position == 1 and entry_price > 0:
            final_bid = float(cleaned.iloc[-1]['bid_price_0'])
            net_diff = (final_bid - entry_price) - (entry_price * (ROUNDTRIP_FEE_BPS / 10000.0))
            realized_pnl += net_diff
            trades += 1
            if net_diff > 0:
                wins += 1

        return {"file": file_path, "trades": trades, "wins": wins, "pnl": realized_pnl}
    except Exception:
        return None

if __name__ == "__main__":
    valid_files = get_verified_parquet_files(r"data\ticks\depth\*\*\*.parquet")
    split = int(len(valid_files) * 0.8)
    test_files = valid_files[split:]

    print("[*] Loading serialized generalized brain...")
    model = PPO.load("models/tachyon_brain_generalized", device="cpu")

    print(f"[*] Evaluating across {len(test_files)} unseen holdout tapes...\n")
    results = []
    for f in test_files:
        res = evaluate_single_file(model, f)
        if res and res["trades"] > 0:
            win_rate = (res["wins"] / res["trades"]) * 100.0
            print(f"[{res['file'].split('symbol=')[-1][:20]:<20}] Trades: {res['trades']:<3} | WinRate: {win_rate:>5.1f}% | Net PnL: ₹{res['pnl']:>7.2f}")
            results.append(res)

    if results:
        tot_trades = sum(r["trades"] for r in results)
        tot_wins = sum(r["wins"] for r in results)
        tot_pnl = sum(r["pnl"] for r in results)
        overall_wr = (tot_wins / tot_trades) * 100.0 if tot_trades > 0 else 0.0
        print("\n" + "="*55)
        print(f"TOTAL TRADES ACROSS HOLDOUT : {tot_trades}")
        print(f"OVERALL WIN RATE            : {overall_wr:.2f}%")
        print(f"CUMULATIVE NET PnL          : ₹{tot_pnl:.2f}")
        print("="*55)
    else:
        print("[!] No trades executed across the holdout set.")
