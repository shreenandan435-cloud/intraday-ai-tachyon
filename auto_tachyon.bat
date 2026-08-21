@echo off
setlocal EnableExtensions

rem ============================================================================
rem  INTRADAY AI TACHYON - unattended launcher (Windows Task Scheduler)
rem
rem  Runs the pre-market boot completely headless: no budget prompt, no key
rem  press, no console to answer. Intended to be fired daily at 09:12 IST, which
rem  is inside NSE's pre-open publication window (09:07-09:15) that the CLAUDE.md
rem  9.3 gap scanner reads.
rem
rem  WHAT --yes BYPASSES, AND WHY THAT IS ACCEPTABLE HERE
rem  ---------------------------------------------------
rem  CLAUDE.md 9.3 calls the budget prompt "the safety-critical screen": it
rem  prints the derived rupee limits and demands a typed confirmation because a
rem  budget wrong by a factor of ten at 09:05 is not recoverable at 15:15.
rem  --yes skips that screen. The mitigation is that the number is no longer
rem  typed under time pressure each morning -- it is fixed in the line below,
rem  reviewed once, and versioned. Change it deliberately, not in a hurry.
rem
rem      SESSION_BUDGET is a RISK BASE, not spending money (CLAUDE.md 1.3):
rem          daily loss limit = SESSION_BUDGET x capital.max_daily_drawdown_pct
rem          per-trade risk   = SESSION_BUDGET x capital.per_trade_risk_pct
rem      At the shipped 2.0%% / 0.4%%, a budget of 10000 gives a 200 rupee daily
rem      loss limit and 40 rupees of risk per trade.
rem
rem  THIS SCRIPT CANNOT TRADE LIVE
rem  -----------------------------
rem  CLAUDE.md 9 requires an interactive confirmation before LIVE. Headless,
rem  that prompt reads EOF, confirm_live() returns False, and the process exits
rem  3 without trading. So a scheduled run is PAPER-only by construction -- if
rem  you ever set TRADING_MODE=LIVE, this task will refuse rather than trade
rem  unattended. That is the fail-safe working, not a bug to route around.
rem
rem  NEVER "End task" FROM TASK SCHEDULER
rem  ------------------------------------
rem  The 15:15 square-off watchdog is a non-daemon thread so it outlives a
rem  graceful shutdown (CLAUDE.md 1.1, 9). A forced termination kills it dead
rem  and can leave a position open overnight. To stop a running session, use
rem  Ctrl-C ONCE in its console, or stop the task and let it drain -- do not
rem  configure "Stop the task if it runs longer than".
rem ============================================================================

rem --- Your default parameters ------------------------------------------------
set "SESSION_BUDGET=10000"

rem Task Scheduler starts in C:\Windows\System32, so every relative path in the
rem project (config\settings.yaml, data\, logs\) would resolve against the wrong
rem directory. Anchor to this file's own folder before anything else.
cd /d "%~dp0"

rem --- Per-day log ------------------------------------------------------------
rem Headless means no console, so unlogged output is lost -- and this process
rem trades money. Locale-independent date via PowerShell: %DATE% is formatted
rem per regional settings and would produce a different filename on a machine
rem set to dd/MM/yyyy than on one set to MM/dd/yyyy.
if not exist "logs" mkdir "logs"
set "TODAY="
for /f "usebackq delims=" %%i in (`powershell -NoProfile -NonInteractive -Command "Get-Date -Format yyyy-MM-dd"`) do set "TODAY=%%i"
if not defined TODAY set "TODAY=undated"
set "LOGFILE=logs\auto_tachyon_%TODAY%.log"

rem --- Locate the virtual environment -----------------------------------------
set "VENV_DIR=.venv"
if not exist "%VENV_DIR%\Scripts\python.exe" set "VENV_DIR=venv"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    >>"%LOGFILE%" echo [%TIME%] FATAL: no virtual environment found in "%CD%".
    >>"%LOGFILE%" echo [%TIME%] Looked for .venv\Scripts\python.exe and venv\Scripts\python.exe
    exit /b 1
)

>>"%LOGFILE%" echo.
>>"%LOGFILE%" echo ============================================================
>>"%LOGFILE%" echo [%DATE% %TIME%] auto_tachyon starting
>>"%LOGFILE%" echo [%TIME%] cwd            : %CD%
>>"%LOGFILE%" echo [%TIME%] venv           : %VENV_DIR%
>>"%LOGFILE%" echo [%TIME%] session budget : %SESSION_BUDGET%
>>"%LOGFILE%" echo ============================================================

rem Activate as requested, so anything that reads VIRTUAL_ENV behaves as it does
rem in an interactive shell. The interpreter is then invoked by its explicit path
rem anyway: activation only prepends to PATH, and calling a bare "python" would
rem still pick up a system interpreter first if activation silently failed.
call "%VENV_DIR%\Scripts\activate.bat"

rem --- Track 1 sidecar: harvest the order book into data\ticks ----------------
rem  Started BEFORE the orchestrator on purpose. The recorder is a ZeroMQ SUB and
rem  the ingestor is the PUB; a subscriber that connects after the publisher has
rem  bound loses whatever was sent inside ZeroMQ's slow-joiner window. Connecting
rem  first costs nothing and closes that hole.
rem
rem  It cannot affect the session: separate process, and the ingestor's PUB socket
rem  sends NOBLOCK and drops rather than waiting on a slow subscriber. Its exit
rem  code is deliberately NOT folded into %RC% below -- a failed harvest is a lost
rem  training sample, not a failed trading day.
rem
rem  --stop-at 15:30 is what makes the shutdown graceful. A Parquet file's footer
rem  is written at close, and Windows cannot deliver a Ctrl+C to a headless child,
rem  so without a self-imposed deadline the only way to stop it is taskkill -- and
rem  that leaves the open file unreadable.
set "RECORDER_LOG=logs\recorder_%TODAY%.log"
start "Tachyon Recorder" /b "%VENV_DIR%\Scripts\python.exe" scripts\run_recorder.py --stop-at 15:30 >>"%RECORDER_LOG%" 2>&1
>>"%LOGFILE%" echo [%TIME%] tick recorder started, logging to %RECORDER_LOG%

rem --- Pre-market boot: scan, write settings.yaml, run the orchestrator --------
rem Blocks here until the session ends (15:30 IST) or the process is stopped.
"%VENV_DIR%\Scripts\python.exe" scripts\boot_tachyon.py --budget %SESSION_BUDGET% --yes >>"%LOGFILE%" 2>&1
set "RC=%ERRORLEVEL%"

rem --- Report the outcome in terms Task Scheduler and a human both understand --
rem  Redirection goes FIRST on every line here, and that is not a style choice.
rem  Written the usual way, `echo ... exit %RC%>>"%LOGFILE%"` expands to
rem  `echo ... exit 0>>"%LOGFILE%"`, and cmd reads a single digit before `>` as a
rem  FILE HANDLE -- so it redirects stdin, prints "exit " with the code eaten, and
rem  logs nothing. Leading redirection cannot be misparsed that way.
if "%RC%"=="0" >>"%LOGFILE%" echo [%TIME%] exit 0 - session ran and shut down cleanly
if "%RC%"=="1" >>"%LOGFILE%" echo [%TIME%] exit 1 - broker login failed; check .env credentials
if "%RC%"=="2" >>"%LOGFILE%" echo [%TIME%] exit 2 - CONFIG FAULT: empty watchlist, bad budget, or an unverifiable token. Not retried by design.
if "%RC%"=="3" >>"%LOGFILE%" echo [%TIME%] exit 3 - ABORTED: booted read-only, or LIVE was not confirmed (expected headless).
if "%RC%"=="4" >>"%LOGFILE%" echo [%TIME%] exit 4 - REFUSED: a session is already running, or today's daily lock is engaged.
>>"%LOGFILE%" echo [%DATE% %TIME%] auto_tachyon finished, exit %RC%

rem Propagate, so Task Scheduler's "Last Run Result" is the real verdict.
endlocal & exit /b %RC%
