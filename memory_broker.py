import sqlite3
import time
from pathlib import Path

DB_PATH = Path("tachyon_state.db")
MAX_STRIKES = 3

def get_connection():
    conn = sqlite3.connect(DB_PATH, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn

def init_ledger():
    with get_connection() as conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS trade_ledger (
                symbol TEXT PRIMARY KEY,
                last_exit_time REAL,
                cooldown_duration REAL DEFAULT 60.0,
                strike_count INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0.0
            )
        ''')

def reset_stale_locks():
    """Flushes stale lockouts from previous sessions while keeping strike history."""
    with get_connection() as conn:
        conn.execute("UPDATE trade_ledger SET last_exit_time = 0")

def log_trade_exit(symbol: str, pnl: float, exit_reason: str):
    current_time = time.time()
    # Tiered cooldown: 60s for Alpha decay/time stops, 300s for hard stop losses
    cooldown = 300.0 if "STOP LOSS" in exit_reason else 60.0
    is_hard_loss = 1 if (pnl < 0 and "STOP LOSS" in exit_reason) else 0

    with get_connection() as conn:
        conn.execute('''
            INSERT INTO trade_ledger (symbol, last_exit_time, cooldown_duration, strike_count, total_pnl)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                last_exit_time = excluded.last_exit_time,
                cooldown_duration = excluded.cooldown_duration,
                strike_count = CASE WHEN ? = 1 THEN trade_ledger.strike_count + 1 ELSE trade_ledger.strike_count END,
                total_pnl = trade_ledger.total_pnl + excluded.total_pnl
        ''', (symbol, current_time, cooldown, is_hard_loss, pnl, is_hard_loss))

def check_trade_gate(symbol: str) -> tuple[bool, str]:
    with get_connection() as conn:
        cursor = conn.execute("SELECT last_exit_time, cooldown_duration, strike_count FROM trade_ledger WHERE symbol = ?", (symbol,))
        row = cursor.fetchone()
        
    if not row:
        return True, "READY"
        
    last_exit, cooldown_dur, strikes = row
    
    if strikes >= MAX_STRIKES:
        return False, f"BLACKLISTED ({strikes} Strikes)"
        
    remaining = cooldown_dur - (time.time() - last_exit)
    if remaining > 0:
        return False, f"COOLING ({int(remaining)}s left)"
        
    return True, "READY"

if __name__ == "__main__":
    init_ledger()
    reset_stale_locks()
    print("[✓] Tachyon Memory Broker upgraded with tiered cooldowns & locks reset.")
