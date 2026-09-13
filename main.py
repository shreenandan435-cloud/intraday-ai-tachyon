import os
import sys
import time
import logging
from config import TachyonConfig
from orchestrator import TachyonOrchestrator
from logger import setup_logger
from telegram_alerter import TelegramAlerter

def pre_session_diagnostic(config: TachyonConfig, mode: str):
    print("--- PRE-SESSION DIAGNOSTIC ---")
    
    # 1. Trading Mode & Interlock
    if mode != "SHADOW_LIVE":
        print(f"[FAIL] Expected TRADING_MODE=SHADOW_LIVE, got {mode}")
        sys.exit(1)
    print(f"[OK] Mode validated: {mode}")
    print("CRITICAL: EXECUTION MODE: SHADOW_LIVE — REAL ORDERS DISABLED")
    
    # 2. API Credentials
    api_key = os.environ.get("SMARTAPI_API_KEY", "")
    if not api_key or len(api_key) < 5:
        print("[FAIL] Missing or invalid SMARTAPI_API_KEY")
        sys.exit(1)
    print("[OK] Angel API credentials located")
    
    # 3. Journal Writable
    journal_path = "shadow_journal.csv"
    try:
        with open(journal_path, 'a') as f:
            f.write("")
        print(f"[OK] Journal path writable: {journal_path}")
    except IOError:
        print(f"[FAIL] Cannot write to {journal_path}")
        sys.exit(1)
        
    # 4. Config & Tokens
    if not config.strategy.active_tokens:
        print("[FAIL] No active tokens configured for subscription")
        sys.exit(1)
    print(f"[OK] Loaded {len(config.strategy.active_tokens)} symbols for subscription")
    
    # 5. Telegram
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    tg_chat = os.environ.get("TELEGRAM_CHAT_ID", "")
    tg_enabled = os.environ.get("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"
    
    if tg_enabled and tg_token and tg_chat:
        print("[OK] Telegram credentials found and alerts enabled")
    else:
        print("[WARN] Telegram alerts disabled or credentials missing")

    print("--- DIAGNOSTIC PASSED ---\n")

def main():
    mode = os.environ.get("TRADING_MODE", "UNKNOWN")
    config = TachyonConfig() 
    
    pre_session_diagnostic(config, mode)
    
    log = setup_logger("Tachyon")
    log.info("Starting Tachyon v2 Engine...")
    
    tg_enabled = os.environ.get("TELEGRAM_ALERTS_ENABLED", "false").lower() == "true"
    tg = TelegramAlerter(
        os.environ.get("TELEGRAM_BOT_TOKEN", ""), 
        os.environ.get("TELEGRAM_CHAT_ID", "")
    )
    
    if tg_enabled:
        tg.send(f"🟢 *Tachyon v2 Initialized*\nMode: `{mode}`\nStatus: Pre-session diagnostic passed. Engines online.")
    
    orchestrator = TachyonOrchestrator(config)
    
    if tg_enabled:
        orchestrator.executor.telegram = tg 
    
    try:
        orchestrator.adapter.start()
        log.info("Awaiting Market Open...")
        
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        log.info("Manual shutdown initiated.")
        if tg_enabled:
            tg.send("🔴 *Tachyon v2 Offline*\nStatus: Manual shutdown.")
    except Exception as e:
        log.critical(f"Unhandled Exception: {e}")
        if tg_enabled:
            tg.send(f"⚠️ *CRITICAL FAILURE*\nException: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
