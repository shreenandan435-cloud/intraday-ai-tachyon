import os
import sys
import time
import subprocess
from datetime import datetime, time as dt_time

def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [SUPERVISOR] {msg}")

def is_market_open():
    now = datetime.now()
    market_open = dt_time(9, 15)
    market_close = dt_time(15, 30)
    return market_open <= now.time() < market_close and now.weekday() < 5

def run_supervisor():
    log("🦅 TACHYON SUPERVISOR ACTIVE")
    engine_process = None

    while True:
        now = datetime.now()
        
        # 1. Weekend Sleep
        if now.weekday() >= 5:
            log("Weekend detected. Sleeping for 1 hour...")
            time.sleep(3600)
            continue

        # 2. Pre-Market Screener (Runs 09:10 to 09:15 IST)
        if dt_time(9, 10) <= now.time() < dt_time(9, 15):
            needs_run = True
            if os.path.exists("resolved_tokens.json"):
                mtime = os.path.getmtime("resolved_tokens.json")
                if (now.timestamp() - mtime) < 1800:
                    needs_run = False

            if needs_run:
                log("Executing Pre-Market Dynamic Screener...")
                subprocess.run([sys.executable, "screener.py"], check=True)
                log("Screener complete. Tokens refreshed.")
            
            time.sleep(20)
            continue

        # 3. Market Hours Execution (09:15 to 15:30 IST)
        if is_market_open():
            if engine_process is None or engine_process.poll() is not None:
                # Failsafe: Run screener if tokens missing or stale (>4 hours)
                if not os.path.exists("resolved_tokens.json") or (now.timestamp() - os.path.getmtime("resolved_tokens.json") > 14400):
                    log("Stale/missing tokens detected. Running screener failsafe...")
                    subprocess.run([sys.executable, "screener.py"], check=True)

                log("Market Open. Launching run_live_session.py subprocess...")
                engine_process = subprocess.Popen([sys.executable, "run_live_session.py"])
            
            time.sleep(5)
            continue

        # 4. Market Post-Close (After 15:30 IST)
        if now.time() >= dt_time(15, 30):
            if engine_process is not None and engine_process.poll() is None:
                log("15:30 IST reached. Terminating Live Session subprocess...")
                engine_process.terminate()
                try:
                    engine_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    engine_process.kill()
                engine_process = None
                log("Live Session cleanly shut down.")

            log("Market closed for today. Sleeping until 09:00 IST tomorrow...")
            time.sleep(300)
            continue

        # 5. Pre-09:10 Morning Standby
        time.sleep(30)

if __name__ == "__main__":
    try:
        run_supervisor()
    except KeyboardInterrupt:
        print("\n[!] Supervisor manually stopped.")
