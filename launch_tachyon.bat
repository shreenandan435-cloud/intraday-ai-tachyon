@echo off
setlocal

rem ============================================================================
rem  INTRADAY AI TACHYON - launcher
rem
rem  Starts the interactive pre-market boot (budget prompt + gap scan, which
rem  rewrites config\settings.yaml and then boots the orchestrator), waits for
rem  you to answer the prompt, then starts the read-only UI terminal.
rem
rem  ORDER MATTERS. The boot sequence rewrites the watchlist, and the UI reads
rem  the watchlist once at startup -- launched first it would spend the session
rem  displaying yesterday's symbols. So this window pauses until you confirm the
rem  budget, and only then brings the UI up.
rem
rem  Each component gets its own console so a fault in one is visible and
rem  killable without touching the other. The UI holds no broker client and
rem  cannot place an order (CLAUDE.md 7.3); closing its window never affects
rem  trading. Closing the Orchestrator window is NOT a graceful shutdown --
rem  use Ctrl-C in that window ONCE, and let it flatten.
rem
rem  To skip the scanner and trade the watchlist already in settings.yaml:
rem      python scripts\boot_tachyon.py --skip-scan
rem  To see what the scan would pick without writing or trading anything:
rem      python scripts\boot_tachyon.py --dry-run
rem ============================================================================

rem Run from the project root regardless of where this was invoked from.
cd /d "%~dp0"

echo.
echo  ###########################################################
echo  #                                                         #
echo  #        I N T R A D A Y   A I   T A C H Y O N            #
echo  #                                                         #
echo  #   NSE intraday  ^|  square-off 15:15 IST  ^|  pre-market  #
echo  #                                                         #
echo  ###########################################################
echo.

rem --- Locate the virtual environment -----------------------------------------
set "VENV_ACTIVATE=.venv\Scripts\activate.bat"
if not exist "%VENV_ACTIVATE%" set "VENV_ACTIVATE=venv\Scripts\activate.bat"
if not exist "%VENV_ACTIVATE%" (
    echo  [FATAL] No virtual environment found.
    echo          Looked for .venv\Scripts\activate.bat and venv\Scripts\activate.bat
    echo          in %CD%
    echo.
    echo          Create one with:
    echo              py -3.14 -m venv .venv
    echo              .venv\Scripts\activate
    echo              pip install -r requirements.txt
    echo              pip install -e .
    echo.
    pause
    exit /b 1
)
echo  [ OK ] venv: %VENV_ACTIVATE%

rem --- 1/4  Track 1 sidecar: harvest the order book into data\ticks -----------
rem  First, and that ordering is the point. The recorder is a ZeroMQ SUB and the
rem  ingestor is the PUB; a subscriber that connects after the publisher has
rem  bound loses everything sent inside ZeroMQ's slow-joiner window.
rem
rem  It cannot touch trading: its own process, and the ingestor's PUB socket sends
rem  NOBLOCK and drops rather than waiting on a slow subscriber. Closing this
rem  window costs you the open Parquet file's footer (the last rotation window),
rem  nothing more -- prefer Ctrl-C ONCE in it.
echo  [ .. ] starting tick recorder ^(Track 1 LOB harvest^)
start "Tachyon Recorder" cmd /k "call %VENV_ACTIVATE% && python scripts\run_recorder.py"

rem --- 2/4  Pre-market boot: budget prompt, gap scan, then the orchestrator ----
rem  This window asks for today's session budget and shows the daily loss limit
rem  CLAUDE.md 1.3 derives from it. Read that screen before typing yes -- the
rem  number is a RISK base, not spending money. In LIVE mode it then asks for a
rem  second, separate confirmation (CLAUDE.md 9).
echo  [ .. ] starting pre-market boot ^(budget prompt + scanner^)
start "Tachyon Orchestrator" cmd /k "call %VENV_ACTIVATE% && python scripts\boot_tachyon.py"

rem --- 3/4  Wait for the config to be written ---------------------------------
echo.
echo  ###########################################################
echo  #  Answer the budget prompt in the Orchestrator window.   #
echo  #  Once it reports the selected watchlist, come back here #
echo  #  and press a key to bring up the UI terminal.           #
echo  ###########################################################
echo.
pause

rem --- 4/4  UI terminal -------------------------------------------------------
echo  [ .. ] starting UI terminal
start "Tachyon UI" cmd /k "call %VENV_ACTIVATE% && python scripts\run_ui.py"

echo  [ .. ] waiting 3s for the UI to bind port 8787
timeout /t 3 /nobreak >nul

echo  [ .. ] opening browser
start "" "http://127.0.0.1:8787"

echo.
echo  [ OK ] Recorder, Orchestrator and UI launched. This one is done.
echo.

endlocal
exit /b 0
