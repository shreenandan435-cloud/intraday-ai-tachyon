@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem --- SOCKS5 Proxy Routing via Giganode SSH Tunnel ---------------------------
rem Routes external outbound traffic (Angel One REST & WebSocket) through the
rem local SOCKS5 tunnel on port 8080. Local bindings (127.0.0.1) are bypassed.
set "HTTP_PROXY=socks5://127.0.0.1:8080"
set "HTTPS_PROXY=socks5://127.0.0.1:8080"
set "ALL_PROXY=socks5://127.0.0.1:8080"
set "NO_PROXY=127.0.0.1,localhost"

rem ---------------------------------------------------------------------------
rem  Auto-start the Giganode SSH tunnel if 127.0.0.1:8080 is not listening.
rem
rem  The SSH dynamic-forward tunnel (-D 8080) is what binds Angel One's WAF
rem  to the whitelisted egress IP 87.76.191.175 — without it, every outbound
rem  call presents the home IP and the broker rejects with HTTP 401/403.
rem  Background processes spawned in one Windows session die with that
rem  session, so a tunnel started manually in a previous console will not
rem  survive the launcher opening fresh windows. The launcher therefore
rem  probes the port first and starts the tunnel in a detached /min window
rem  when it is down. ``-N`` prevents the shell from being held open; ``-q``
rem  suppresses the banner; ``-o ExitOnForwardFailure=yes`` ensures the
rem  client dies immediately when the forward can't bind rather than
rem  silently sitting on a half-open port.
rem ---------------------------------------------------------------------------
netstat -ano | findstr :8080 | findstr LISTENING >nul
if errorlevel 1 (
    echo  [ .. ] port 8080 not listening — spawning Giganode SSH SOCKS5 tunnel...
    start "Giganode SOCKS5 Tunnel" /min ssh -D 8080 -N -q -o ExitOnForwardFailure=yes root@87.76.191.175
    rem Give ssh.exe ~3 s to bind the SOCKS listener before the engine starts
    rem firing requests; without this gap the first request races the bind
    rem and surfaces as a transient WinError 10061.
    timeout /t 3 /nobreak >nul
    netstat -ano | findstr :8080 | findstr LISTENING >nul
    if errorlevel 1 (
        echo  [FAIL] SSH tunnel failed to bind 127.0.0.1:8080 within 3 s.
        echo         Check that ssh.exe is on PATH and that your key is loaded
        echo         for root@87.76.191.175. Run scripts/diagnose_proxy.py for
        echo         a full handshake trace.
        pause
        exit /b 1
    )
    echo  [ OK ] SSH tunnel listening on 127.0.0.1:8080
) else (
    echo  [ OK ] socks proxy  : 127.0.0.1:8080 (already listening — reused)
)

rem ============================================================================
rem  INTRADAY AI TACHYON - Step 7 paper-session launcher
rem
rem  Single-click boot for the modernized trading system. Launches the Step 7
rem  live engine (scripts/run_live_session.py) and the read-only observability
rem  dashboard (scripts/run_ui.py) in two INDEPENDENT console windows. The
rem  launching console itself returns control to the operator immediately --
rem  closing it does NOT stop either child.
rem
rem  Order of operations:
rem    1. cd to the project root
rem    2. print startup banner + diagnostics (python, user, host)
rem    3. KILL any lingering python.exe (legacy-bot zombie guard)
rem       3b. 1s grace period for OS handle release (timeout /t 1)
rem    4. activate the virtual environment
rem    5. set PYTHONPATH=%cd%
rem    6. verify the two entry points exist
rem    7. start "Tachyon - Dashboard"    : python scripts/run_ui.py
rem       (UI first so its HTTP port is bound before the browser auto-launch)
rem    8. timeout /t 2 /nobreak >nul     : give the dashboard time to bind
rem    9. start http://127.0.0.1:8787    : auto-open the browser to the dashboard
rem   10. start "Tachyon - Core Engine"  : python scripts/run_live_session.py
rem   11. echo final banner + dashboard URL, exit 0
rem ============================================================================

rem --- 1/11 project root -------------------------------------------------------
cd /d "C:\Users\Shree\intraday-ai-tachyon"
if errorlevel 1 (
    echo  [FAIL] Project root not found: C:\Users\Shree\intraday-ai-tachyon
    echo         Update the cd /d path at the top of this file.
    pause
    exit /b 1
)

echo.
echo  ###########################################################
echo  #   I N T R A D A Y   A I   T A C H Y O N                 #
echo  #   Step 7 live engine + observability dashboard          #
echo  ###########################################################
echo.

rem --- 2/11 diagnostics ------------------------------------------------------
echo  [ OK ] project root : %CD%
for /f "delims=" %%P in ('where python 2^>nul') do (
    set "PYTHON_BIN=%%P"
    goto :got_python
)
set "PYTHON_BIN=<not on PATH>"
:got_python
echo  [ OK ] python       : !PYTHON_BIN!
echo  [ OK ] user / host  : %USERNAME% @ %COMPUTERNAME%
echo  [ OK ] os           : %OS%  session: %SESSIONNAME%
echo  [ OK ] socks proxy  : %HTTP_PROXY% (bypassing %NO_PROXY%)
echo.

rem --- 3/11 kill lingering python.exe (legacy-bot zombie guard) --------------
set "PY_COUNT=0"
for /f %%N in ('tasklist /FI "IMAGENAME eq python.exe" 2^>nul ^| find /I "python.exe" ^| find /C "python.exe"') do set "PY_COUNT=%%N"
if !PY_COUNT! GTR 0 (
    echo  [ .. ] killing !PY_COUNT! lingering python.exe process^(es^)
    echo         ^(this ends any legacy bot instance that may still be running^)
    taskkill /F /IM python.exe /T 2>nul
    timeout /t 1 /nobreak >nul
    set "PY_AFTER=0"
    for /f %%N in ('tasklist /FI "IMAGENAME eq python.exe" 2^>nul ^| find /I "python.exe" ^| find /C "python.exe"') do set "PY_AFTER=%%N"
    if !PY_AFTER! GTR 0 (
        echo.
        echo  [FAIL] !PY_AFTER! python.exe process^(es^) survived the kill.
        echo         End them in Task Manager and re-run this launcher.
        echo.
        pause
        exit /b 1
    )
    echo  [ OK ] lingering python.exe terminated
) else (
    echo  [ OK ] no lingering python.exe
)
echo.

rem --- 4/11 activate the virtual environment ----------------------------------
set "VENV_ACTIVATE=.venv\Scripts\activate.bat"
if not exist "%VENV_ACTIVATE%" set "VENV_ACTIVATE=venv\Scripts\activate.bat"
if not exist "%VENV_ACTIVATE%" (
    echo  [FAIL] No virtual environment found.
    echo         Looked for .venv\Scripts\activate.bat and venv\Scripts\activate.bat
    echo         in %CD%
    echo         Create one with:
    echo             py -3.14 -m venv .venv
    echo             .venv\Scripts\activate
    echo             pip install -r requirements.txt
    echo             pip install -e .
    echo.
    pause
    exit /b 1
)
call "%VENV_ACTIVATE%"
if errorlevel 1 (
    echo  [FAIL] Could not activate the virtual environment ^(error %errorlevel%^).
    pause
    exit /b 1
)
echo  [ OK ] venv active : %VENV_ACTIVATE%
echo.

rem --- 5/11 PYTHONPATH so package imports resolve from cwd ------------------
set "PYTHONPATH=%cd%"
echo  [ OK ] PYTHONPATH  : %PYTHONPATH%
echo.

rem --- 6/11 verify entry points ---------------------------------------------
set "ENGINE_SCRIPT=scripts\run_live_session.py"
set "DASH_SCRIPT=scripts\run_ui.py"
set "ENTRY_OK=1"
if not exist "%ENGINE_SCRIPT%" (
    echo  [FAIL] missing entry point: %ENGINE_SCRIPT%
    set "ENTRY_OK=0"
) else (
    echo  [ OK ] engine      : %ENGINE_SCRIPT%
)
if not exist "%DASH_SCRIPT%" (
    echo  [FAIL] missing entry point: %DASH_SCRIPT%
    set "ENTRY_OK=0"
) else (
    echo  [ OK ] dashboard   : %DASH_SCRIPT%
)
if "!ENTRY_OK!"=="0" (
    echo.
    pause
    exit /b 1
)
echo.

rem --- 7/11 start the dashboard FIRST (so its port is up for the browser) ----
echo  [ .. ] launching dashboard   :  Tachyon - Dashboard
start "Tachyon - Dashboard" cmd /k "python scripts\run_ui.py"

rem --- 8/11 give the dashboard a moment to bind to its port ------------------
timeout /t 2 /nobreak >nul

rem --- 9/11 auto-open the browser to the dashboard ---------------------------
if not errorlevel 1 (
    start http://127.0.0.1:8787
)

rem --- 10/11 start the core engine -------------------------------------------
echo  [ .. ] launching core engine:  Tachyon - Core Engine
start "Tachyon - Core Engine" cmd /k "python scripts\run_live_session.py --mode paper --log-every 50"

rem --- 11/11 final banner ----------------------------------------------------
echo.
echo  ###########################################################
echo  # Both consoles are running independently.                #
echo  #   - "Tachyon - Core Engine" : the trading engine         #
echo  #   - "Tachyon - Dashboard"   : the observability terminal #
echo  #                                                         #
echo  # Dashboard URL : http://127.0.0.1:8787                   #
echo  #                                                         #
echo  # Closing THIS window does NOT stop either child.         #
echo  # Close the "Tachyon - Core Engine" window to stop the    #
echo  # trading session gracefully.                             #
echo  ###########################################################
echo.

endlocal & exit /b 0