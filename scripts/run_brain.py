"""Process B entrypoint — the strategy brain (CLAUDE.md §2, §9).

Run::

    .venv\\Scripts\\python.exe scripts/run_brain.py

Consumes the tick spine published by ``run_ingestor.py``, computes indicators, produces
signals, and — only with the Risk Engine's permission — places Robo orders.

``TRADING_MODE`` defaults to PAPER. **LIVE requires the env var set explicitly *and* an
interactive confirmation here**, because a process that can go live by accident will.
"""

from __future__ import annotations

import asyncio
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tachyon.core import eventloop  # noqa: E402
from tachyon.core.config import Settings, get_settings  # noqa: E402
from tachyon.core.constants import TradingMode  # noqa: E402
from tachyon.core.logger import configure_logging, get_logger, shutdown_logging  # noqa: E402
from tachyon.execution.api import SmartApiClient, SmartApiError  # noqa: E402
from tachyon.strategy.brain import StrategyBrain  # noqa: E402


def _confirm_live(settings: Settings) -> bool:
    """Require a typed confirmation before trading real money (CLAUDE.md §9).

    Reads from the terminal, not from a flag or an env var. Both of those persist, and a
    persistent "yes I meant LIVE" is indistinguishable from a forgotten one.
    """
    if settings.trading_mode is not TradingMode.LIVE:
        return True

    settings.validate_live_ready()
    print("\n" + "=" * 72)
    print("  TRADING_MODE=LIVE — this session will place REAL ORDERS with REAL MONEY.")
    print(f"  Daily loss limit: Rs.500 (latching)   Watchlist: {len(settings.watchlist)} symbols")
    print("  Auto square-off at 15:15 IST is unconditional.")
    print("=" * 72)
    try:
        answer = input("  Type LIVE to confirm, anything else to abort: ").strip()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer == "LIVE"


def _install_signal_handlers(brain: StrategyBrain) -> None:
    """Request a graceful shutdown on SIGINT/SIGTERM (CLAUDE.md §9).

    ``loop.add_signal_handler`` is POSIX-only, so fall back to ``signal.signal`` on Windows.
    """
    loop = asyncio.get_running_loop()

    def request_stop() -> None:
        loop.create_task(brain.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, request_stop)
        except (NotImplementedError, AttributeError):
            signal.signal(sig, lambda _s, _f: request_stop())


async def _build_client(settings: Settings) -> SmartApiClient | None:
    """Log in, or return ``None`` in PAPER when no credentials are configured.

    In LIVE a login failure is fatal: the reconciler would otherwise be unable to prove the
    account is flat, lock the session, and leave the operator debugging a lockdown whose real
    cause was a bad TOTP.
    """
    log = get_logger("run_brain")
    if not settings.smartapi_password.get_secret_value():
        if settings.trading_mode is TradingMode.LIVE:
            raise SmartApiError("login", "LIVE mode with no SMARTAPI_PASSWORD configured")
        log.warning(
            "brain.no_broker_client",
            mode=settings.trading_mode,
            impact="PAPER simulation only; reconciliation will be skipped",
        )
        return None

    client = SmartApiClient(settings=settings)
    await client.login()
    return client


async def _main() -> int:
    settings = get_settings()
    configure_logging(
        role="brain",
        level=settings.log_level,
        log_dir=settings.log_dir,
        mode=settings.trading_mode,
    )
    log = get_logger("run_brain")

    if not settings.watchlist:
        log.critical("brain.empty_watchlist", hint="add symbols to config/settings.yaml")
        return 2

    if not _confirm_live(settings):
        log.critical("brain.live_not_confirmed", action="aborting without trading")
        return 3

    client: SmartApiClient | None = None
    try:
        client = await _build_client(settings)
    except SmartApiError as exc:
        log.critical("brain.login_failed", error=str(exc))
        shutdown_logging()
        return 1

    brain = StrategyBrain(settings=settings, client=client)
    _install_signal_handlers(brain)

    try:
        tradeable = await brain.boot()
        if not tradeable:
            log.critical(
                "brain.read_only",
                state=brain.state,
                hint="daily lock engaged, or the broker holds exposure this process did not "
                "create — flatten it manually and restart",
            )
        # The loop runs either way. A locked session still needs the 15:15 watchdog armed and
        # telemetry flowing; what it does not get is the ability to place an order.
        await brain.run()
    except asyncio.CancelledError:
        log.info("brain.cancelled")
    finally:
        if client is not None:
            await client.aclose()
        shutdown_logging()
    return 0


if __name__ == "__main__":
    try:
        # eventloop.run, not asyncio.run: on Windows the default Proactor loop cannot host a
        # zmq.asyncio socket at all (see tachyon.core.eventloop).
        raise SystemExit(eventloop.run(_main()))
    except KeyboardInterrupt:
        raise SystemExit(130) from None
