import os
import sys
import json
import glob
import urllib.request
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_BOT_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID")
TELEGRAM_ALERTS_ENABLED = os.getenv("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"

def send_telegram(text: str):
    if not TELEGRAM_ALERTS_ENABLED or not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = json.dumps({"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5):
            pass
    except Exception as e:
        print(f"[!] Telegram alert failed: {e}")

def run_forensic_audit(target_date: str = None):
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    log_path = f"logs/session_{target_date}.jsonl"
    if not os.path.exists(log_path):
        print(f"[!] No log file found at: {log_path}")
        return

    trades = []
    latencies = []

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("event") == "TRADE_EXIT":
                    trades.append(rec)
                elif rec.get("event") == "LATENCY_SAMPLE":
                    latencies.append(rec)
            except Exception:
                continue

    print("\n" + "=" * 75)
    print(f"       TACHYON POST-MARKET FORENSIC AUDIT — {target_date}")
    print("=" * 75)

    # 1. Latency Profile
    q_lats = [l["queue_latency_us"] for l in latencies]
    inf_lats = [l["infer_latency_us"] for l in latencies]

    p50_q = np.percentile(q_lats, 50) if q_lats else 0.0
    p95_q = np.percentile(q_lats, 95) if q_lats else 0.0
    p99_q = np.percentile(q_lats, 99) if q_lats else 0.0

    p50_inf = np.percentile(inf_lats, 50) if inf_lats else 0.0
    p95_inf = np.percentile(inf_lats, 95) if inf_lats else 0.0
    p99_inf = np.percentile(inf_lats, 99) if inf_lats else 0.0

    print("\n--- 1. INGESTION & INFERENCE LATENCY BENCHMARKS ---")
    print(f"• Total Sampled Ticks   : {len(latencies) * 500:,}")
    print(f"• Queue Wait Time (P50) : {p50_q:6.1f} µs | (P95): {p95_q:6.1f} µs | (P99): {p99_q:6.1f} µs")
    print(f"• CPU Inference  (P50)  : {p50_inf:6.1f} µs | (P95): {p95_inf:6.1f} µs | (P99): {p99_inf:6.1f} µs")
    print(f"• Total End-to-End Latency: ~{(p50_q + p50_inf) / 1000.0:.2f} ms (sub-millisecond target met: {(p50_q + p50_inf) < 1000.0})")

    # 2. Trade PnL & Financial Profile
    if not trades:
        print("\n[!] Zero trades were executed during this session.")
        send_telegram(
            f"📊 *TACHYON POST-MARKET AUDIT* ({target_date})\n"
            f"• Trades: `0` (Strict Cash Filter Kept System Flat)\n"
            f"• Ingestion Latency (P50): `{p50_q:.1f} µs`\n"
            f"• Inference Latency (P50): `{p50_inf:.1f} µs`"
        )
        return

    df = pd.DataFrame(trades)
    tot_trades = len(df)
    wins = df[df["net_pnl"] > 0]
    losses = df[df["net_pnl"] <= 0]
    win_rate = (len(wins) / tot_trades) * 100.0
    net_pnl = df["net_pnl"].sum()
    gross_pnl = df["gross_pnl"].sum()
    fees_paid = df["fee"].sum()

    gross_wins = wins["gross_pnl"].sum() if not wins.empty else 0.0
    gross_losses = abs(losses["gross_pnl"].sum()) if not losses.empty else 0.0
    profit_factor = (gross_wins / gross_losses) if gross_losses > 0 else np.inf

    avg_win = wins["net_pnl"].mean() if not wins.empty else 0.0
    avg_loss = losses["net_pnl"].mean() if not losses.empty else 0.0
    expectancy = (win_rate / 100.0 * avg_win) + ((1 - win_rate / 100.0) * avg_loss)

    print("\n--- 2. TRADE EXECUTION & FINANCIAL PERFORMANCE ---")
    print(f"• Executed Trades       : {tot_trades}")
    print(f"• Win Rate              : {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"• Realized Net PnL      : ₹{net_pnl:+8.2f} (Gross: ₹{gross_pnl:+.2f} | Fees: ₹{fees_paid:.2f})")
    print(f"• Profit Factor         : {profit_factor:8.2f}")
    print(f"• Expectancy per Trade  : ₹{expectancy:+8.2f}")
    print(f"• Avg Win vs Avg Loss   : ₹{avg_win:+.2f} vs ₹{avg_loss:+.2f}")

    # 3. Execution Quality & Stop-Loss Slippage
    sl_trades = df[df["reason"].str.contains("STOP-LOSS")]
    avg_sl_slippage = sl_trades["slippage_bps"].mean() if not sl_trades.empty else 0.0
    avg_entry_spread = df["entry_spread_bps"].mean()
    med_hold_sec = df["hold_seconds"].median()
    max_hold_sec = df["hold_seconds"].max()

    print("\n--- 3. MICROSTRUCTURE SLIPPAGE & EXECUTION INTEGRITY ---")
    print(f"• Average Entry Spread  : {avg_entry_spread:.2f} bps")
    print(f"• Median Trade Duration : {med_hold_sec:.1f}s (Max: {max_hold_sec:.1f}s)")
    print(f"• Stop-Loss Triggers    : {len(sl_trades)} occurrences")
    if not sl_trades.empty:
        print(f"• Stop-Loss Slippage    : {avg_sl_slippage:+.2f} bps (vs -20.0 bps hard target)")
    else:
        print("• Stop-Loss Slippage    : None triggered (All closed via Alpha Exit or EOD Square-Off)")

    # 4. Symbol Attribution Table
    print("\n--- 4. PER-SYMBOL ATTRIBUTION MATRIX ---")
    print(f"{'Symbol':<12} | {'Trades':<6} | {'Win Rate':<8} | {'Net PnL':<10} | {'Avg Spread':<10}")
    print("-" * 55)
    symbol_md_rows = []
    for sym, g in df.groupby("symbol"):
        sym_trades = len(g)
        sym_wins = len(g[g["net_pnl"] > 0])
        sym_wr = (sym_wins / sym_trades) * 100.0
        sym_pnl = g["net_pnl"].sum()
        sym_spread = g["entry_spread_bps"].mean()
        print(f"{sym:<12} | {sym_trades:<6} | {sym_wr:>6.1f}% | ₹{sym_pnl:>8.2f} | {sym_spread:>8.1f} bps")
        symbol_md_rows.append(f"| `{sym}` | {sym_trades} | {sym_wr:.1f}% | ₹{sym_pnl:+.2f} | {sym_spread:.1f} bps |")

    # 5. Save Markdown Audit Report
    md_report_path = f"logs/audit_{target_date}.md"
    md_content = f"""# Tachyon Forensic Audit Report — {target_date}

## 1. Latency Profile
- **Sampled Ticks**: {len(latencies) * 500:,}
- **Queue Wait Time**: P50: `{p50_q:.1f} µs` | P95: `{p95_q:.1f} µs` | P99: `{p99_q:.1f} µs`
- **Inference Compute**: P50: `{p50_inf:.1f} µs` | P95: `{p95_inf:.1f} µs` | P99: `{p99_inf:.1f} µs`
- **Total Decision Latency**: `~{(p50_q + p50_inf) / 1000.0:.2f} ms`

## 2. Financial Metrics
- **Total Trades**: {tot_trades}
- **Win Rate**: `{win_rate:.1f}%` ({len(wins)}W / {len(losses)}L)
- **Net Realized PnL**: `₹{net_pnl:+.2f}`
- **Profit Factor**: `{profit_factor:.2f}`
- **Expectancy**: `₹{expectancy:+.2f} per trade`

## 3. Microstructure & Slippage
- **Average Entry Spread**: `{avg_entry_spread:.2f} bps`
- **Stop-Loss Ejections**: `{len(sl_trades)}`
- **Stop-Loss Slippage**: `{avg_sl_slippage:+.2f} bps` (vs -20 bps daemon target)
- **Median Hold Time**: `{med_hold_sec:.1f}s`

## 4. Symbol Attribution
| Symbol | Trades | Win Rate | Net PnL | Avg Spread |
| :--- | :--- | :--- | :--- | :--- |
""" + "\n".join(symbol_md_rows)

    with open(md_report_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"\n[✓] Markdown audit saved to: {md_report_path}")

    # 6. Telegram Executive Summary Dispatch
    tg_summary = (
        f"📋 *TACHYON POST-MARKET AUDIT* ({target_date})\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"• *Net Realized PnL*: `₹{net_pnl:+.2f}`\n"
        f"• *Trades*: `{tot_trades}` | *Win Rate*: `{win_rate:.1f}%`\n"
        f"• *Profit Factor*: `{profit_factor:.2f}`\n"
        f"• *Avg Spread Paid*: `{avg_entry_spread:.1f} bps`\n"
        f"• *Stop-Loss Slippage*: `{avg_sl_slippage:+.2f} bps`\n"
        f"• *Decision Latency*: `{(p50_q + p50_inf)/1000.0:.2f} ms` (P50)\n"
        f"━━━━━━━━━━━━━━━━━━━━━\n"
        f"Full report saved to `{md_report_path}`"
    )
    send_telegram(tg_summary)

if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else None
    run_forensic_audit(target)
