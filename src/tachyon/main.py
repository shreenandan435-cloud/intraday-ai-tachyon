"""The orchestrator — boots and supervises the whole system, CLAUDE.md §2, §9.

One command starts everything::

    .venv\\Scripts\\python.exe -m tachyon.main

Headless paper trading mode:
- No interactive prompts
- All config from environment variables / .env
- Paper trading engine with virtual capital
- Telemetry logging (RotatingFileHandler, CSV, Parquet)
- Telegram alerts
- Graceful SIGTERM/SIGINT handling
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import sys
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable, Final, Optional

from tachyon.core import eventloop
from tachyon.core.clock import SYSTEM_CLOCK
from tachyon.core.config import Settings, get_settings
from tachyon.core.constants import PROJECT_ROOT, TradingMode
from tachyon.core.logger import configure_logging, get_logger, shutdown_logging
from tachyon.core.shutdown import (
    GracefulShutdown,
    close_client_sessions,
    flush_all_logs,
    install_signal_handlers,
    notify_systemd_ready,
    notify_systemd_stopping,
)
from tachyon.execution.api import SmartApiClient, SmartApiError
from tachyon.paper import PaperTradingEngine, create_paper_engine
from tachyon.strategy.brain import StrategyBrain, _SymbolState
from tachyon.telemetry import TelemetryManager, create_telemetry_manager

_log = get_logger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

#: The ingestor entrypoint, launched as a child process.
INGESTOR_SCRIPT: Final[Path] = PROJECT_ROOT / "scripts" / "run_ingestor.py"

#: How many times the ingestor may be restarted before we stop trying.
MAX_INGESTOR_RESTARTS: Final[int] = 5

#: A child that survives this long is treated as healthy, and the restart counter resets.
STABLE_CHILD_SECONDS: Final[float] = 60.0

#: Exit codes the ingestor uses for a *configuration* fault.
CONFIG_FAULT_EXIT_CODES: Final[frozenset[int]] = frozenset({2})

#: How long the ingestor gets to exit politely before it is killed.
TERMINATE_GRACE_SECONDS: Final[float] = 10.0

#: How long shutdown waits for the Brain.
BRAIN_SHUTDOWN_SECONDS: Final[float] = 30.0


@dataclass(slots=True)
class SupervisorStats:
    """Counters for observability."""

    ingestor_starts: int = 0
    ingestor_restarts: int = 0
    ingestor_exit_code: int | None = None
    gave_up_on_ingestor: bool = False


class IngestorSupervisor:
    """Runs ``scripts/run_ingestor.py`` as a child process and keeps it alive.

    Args:
        script: entrypoint to launch.
        max_restarts: restart budget for the session.
        enabled: False leaves ingestion to the operator (useful when it is already running).
    """

    __slots__ = ("_enabled", "_max_restarts", "_process", "_script", "_stopping", "stats")

    def __init__(
        self,
        *,
        script: Path = INGESTOR_SCRIPT,
        max_restarts: int = MAX_INGESTOR_RESTARTS,
        enabled: bool = True,
    ) -> None:
        self._script = script
        self._max_restarts = max_restarts
        self._enabled = enabled
        self._process: asyncio.subprocess.Process | None = None
        self._stopping = asyncio.Event()
        self.stats = SupervisorStats()

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    async def _spawn(self) -> asyncio.subprocess.Process:
        """Launch the child, inheriting stdout/stderr so its logs land beside ours."""
        self.stats.ingestor_starts += 1
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self._script),
            cwd=str(PROJECT_ROOT),
        )
        _log.info("supervisor.ingestor_started", pid=process.pid, script=str(self._script))
        return process

    async def run(self) -> None:
        """Keep the ingestor alive until :meth:`stop`. Never raises."""
        if not self._enabled:
            _log.warning(
                "supervisor.ingestor_disabled",
                impact="no market data unless run_ingestor.py is already running elsewhere",
            )
            return
        if not self._script.is_file():
            _log.critical(
                "supervisor.ingestor_missing",
                script=str(self._script),
                impact="NO MARKET DATA — the Brain will flag FEED_STALE and refuse entries",
            )
            return

        restarts = 0
        while not self._stopping.is_set():
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                self._process = await self._spawn()
                code = await self._process.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                code = -1
                _log.critical(
                    "supervisor.spawn_failed", error=str(exc), error_type=type(exc).__name__
                )

            self.stats.ingestor_exit_code = code
            if self._stopping.is_set():
                _log.info("supervisor.ingestor_exited", code=code, reason="shutdown")
                return

            if code in CONFIG_FAULT_EXIT_CODES:
                self.stats.gave_up_on_ingestor = True
                _log.critical(
                    "supervisor.ingestor_misconfigured",
                    exit_code=code,
                    impact="NO MARKET DATA. Not retrying — a configuration fault does not "
                    "heal itself. Check .env and config/settings.yaml, then restart.",
                )
                return

            uptime = loop.time() - started
            if uptime >= STABLE_CHILD_SECONDS:
                restarts = 0
            restarts += 1
            self.stats.ingestor_restarts += 1

            if restarts > self._max_restarts:
                self.stats.gave_up_on_ingestor = True
                _log.critical(
                    "supervisor.ingestor_gave_up",
                    restarts=restarts - 1,
                    impact="NO MARKET DATA for the rest of the session. The Brain will refuse "
                    "every entry on FEED_STALE and the 15:15 square-off still runs.",
                )
                return

            _log.critical(
                "supervisor.ingestor_restarting",
                exit_code=code,
                attempt=restarts,
                uptime_seconds=round(uptime, 1),
                cost="per-token sequence restarts at 1, so the Brain will see a gap and "
                "SESSION VWAP IS VOID for the rest of the day — no new signals",
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=2.0)

    async def stop(self) -> None:
        """Terminate the child and wait for it, escalating to a kill if it will not go."""
        self._stopping.set()
        process = self._process
        if process is None or process.returncode is not None:
            return

        _log.info("supervisor.terminating_ingestor", pid=process.pid)
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            _log.error(
                "supervisor.killing_ingestor",
                pid=process.pid,
                waited_seconds=TERMINATE_GRACE_SECONDS,
            )
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(TimeoutError, Exception):
                await asyncio.wait_for(process.wait(), timeout=5.0)


class Orchestrator:
    """Boots the ingestor, paper engine, brain, and telemetry.

    Args:
        settings: resolved config.
        client: broker client (None for paper mode).
        supervisor: injected for tests.
        brain: injected for tests.
        paper_engine: injected for tests.
        telemetry: injected for tests.
    """

    __slots__ = (
        "_brain",
        "_client",
        "_paper_engine",
        "_settings",
        "_shutdown",
        "_supervisor",
        "_telemetry",
        "_tasks",
    )

    def __init__(
        self,
        *,
        settings: Settings,
        client: SmartApiClient | None = None,
        supervisor: IngestorSupervisor | None = None,
        brain: StrategyBrain | None = None,
        paper_engine: "PaperTradingEngine | None" = None,
        telemetry: "TelemetryManager | None" = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._supervisor = supervisor if supervisor is not None else IngestorSupervisor()
        self._brain = brain
        self._paper_engine = paper_engine
        self._telemetry = telemetry
        self._tasks: list[asyncio.Task[None]] = []
        self._shutdown: GracefulShutdown | None = None

    @property
    def brain(self) -> StrategyBrain | None:
        return self._brain

    @property
    def supervisor(self) -> IngestorSupervisor:
        return self._supervisor

    @property
    def paper_engine(self) -> "PaperTradingEngine | None":
        return self._paper_engine

    @property
    def telemetry(self) -> "TelemetryManager | None":
        return self._telemetry

    def install_signal_handlers(self) -> None:
        """Install signal handlers using GracefulShutdown."""
        if self._shutdown:
            # Signal handlers are installed in __aenter__
            pass

    async def run(self) -> int:
        """Boot everything, run until shutdown, and tear down in order."""
        # Initialize components
        await self._initialize()

        # Boot the brain (runs boot sequence, returns whether trading is allowed)
        tradeable = True
        if self._brain:
            tradeable = await self._brain.boot()
            if not tradeable:
                _log.critical(
                    "orchestrator.read_only",
                    state=self._brain.state if hasattr(self._brain, 'state') else "UNKNOWN",
                    hint="daily lock engaged, or the broker holds exposure this process did not "
                    "create — flatten it manually and restart",
                )

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._supervisor.run(), name="ingestor-supervisor"),
            asyncio.create_task(self._periodic_telemetry(), name="periodic-telemetry"),
        ]

        # Start brain if available (always run for telemetry and watchdog, even in read-only mode)
        if self._brain:
            self._tasks.append(asyncio.create_task(self._brain.run(), name="brain-run"))

        # Start paper engine telemetry
        if self._paper_engine:
            self._tasks.append(asyncio.create_task(self._paper_engine_telemetry(), name="paper-telemetry"))

        _log.info(
            "orchestrator.running",
            mode=self._settings.trading_mode.value,
            state=self._brain.state if self._brain else "NO_BRAIN",
            symbols=len(self._settings.watchlist),
            ingestor_pid=self._supervisor.pid,
            paper_engine=self._paper_engine is not None,
        )

        try:
            # Wait for shutdown signal
            if self._shutdown:
                exit_code = await self._shutdown.wait_for_shutdown()
            else:
                # Fallback: wait for any task to complete
                done, pending = await asyncio.wait(
                    self._tasks,
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                exit_code = 0
        finally:
            await self._teardown()

        # Return exit code 3 for read-only (locked) sessions that still ran
        if self._brain and not tradeable:
            return 3
        return exit_code

    async def request_shutdown(self, reason: str = "manual") -> None:
        """Request shutdown from external caller (e.g., test)."""
        if self._shutdown:
            self._shutdown.request_shutdown(reason)

    async def _initialize(self) -> None:
        """Initialize all components in order."""
        # Initialize telemetry first
        self._telemetry = create_telemetry_manager()
        self._telemetry.log_operation("orchestrator_initializing")

        # Initialize paper trading engine
        self._paper_engine = create_paper_engine(self._settings)
        await self._paper_engine.start()

        # Initialize brain if in live mode or paper with brain
        if self._settings.trading_mode == TradingMode.LIVE:
            raise RuntimeError("LIVE mode not supported in headless paper trading mode")

        # For paper trading, we still create a Brain for signal generation
        # but it will use the paper engine for execution
        from tachyon.execution.api import SmartApiClient
        from tachyon.execution.builder import OrderBuilder
        from tachyon.execution.executor import RoboExecutor
        from tachyon.execution.journal import OrderJournal
        from tachyon.execution.reconciliation import OrderBookPoller, StateReconciler
        from tachyon.ipc.monitor import FeedMonitor
        from tachyon.ipc.subscriber import AsyncSubscriber, SubscriberRole
        from tachyon.ui.postback import OrderStatusListener, watchlist_resolver
        from tachyon.math_engine.core import TickAggregator
        from tachyon.math_engine.warmup import warmup
        from tachyon.persistence.trade_logger import TradeLogger
        from tachyon.risk.budget import SessionBudget
        from tachyon.risk.engine import RiskEngine
        from tachyon.risk.tracker import PnLTracker, PositionRegistry
        from tachyon.risk.watchdog import SquareOffWatchdog
        from tachyon.sentinel.service import SentinelDaemon
        from tachyon.sentinel.state import MacroSnapshot, MacroState
        from tachyon.strategy.brain import BrainStats
        from tachyon.strategy.cooldown import ReentryManager
        from tachyon.strategy.signals import SignalGenerator
        from tachyon.strategy.telemetry import StatePublisher
        from tachyon.utils.telegram_alerts import TelegramAlerter
        from tachyon.core.state import DailyLock, StateMachine

        # Initialize core components for Brain
        lock = DailyLock()
        state_machine = StateMachine.boot(lock)
        from datetime import timedelta

        budget = SessionBudget.resolve(
            capital=self._settings.capital.session_budget_inr,
            drawdown_pct=self._settings.capital.max_daily_drawdown_pct,
            per_trade_pct=self._settings.capital.per_trade_risk_pct,
        )
        budget.log_summary()

        # Create brain components only if not already provided
        if self._brain is None:
            self._brain = StrategyBrain(
                settings=self._settings,
                client=None,  # Paper mode - no broker client
                state_machine=state_machine,
                trade_logger=TradeLogger(),
            )
        else:
            # Use the injected brain, but initialize its components for paper trading
            # Provide a default clock if the brain doesn't have one (for test fakes)
            if not hasattr(self._brain, '_clock') or self._brain._clock is None:
                self._brain._clock = SYSTEM_CLOCK
            
            self._brain._settings = self._settings
            self._brain._state = state_machine
            self._brain._journal = TradeLogger()
            self._brain._budget = SessionBudget.resolve(
                capital=self._settings.capital.session_budget_inr,
                drawdown_pct=self._settings.capital.max_daily_drawdown_pct,
                per_trade_pct=self._settings.capital.per_trade_risk_pct,
            )
            self._brain._budget.log_summary()
            self._brain._pnl = PnLTracker(
                self._brain._state,
                daily_lock=DailyLock(),
                mode=self._settings.trading_mode,
                clock=self._brain._clock,
                limit=self._brain._budget.daily_loss_limit,
                on_trip=getattr(self._brain, '_on_kill_switch', lambda e: None),
            )
            self._brain._positions = PositionRegistry()
            self._brain._monitor = FeedMonitor(clock=self._brain._clock)
            self._brain._macro = MacroState(
                risk_off_confidence=self._settings.sentinel.risk_off_confidence,
                blacklist_window=timedelta(minutes=self._settings.sentinel.news_blacklist_minutes),
                clock=self._brain._clock,
            )
            self._brain._sentinel = SentinelDaemon(state=self._brain._macro, settings=self._settings, clock=self._brain._clock)
            self._brain._risk = RiskEngine(
                self._brain._state,
                self._brain._pnl,
                self._brain._monitor,
                self._brain._positions,
                settings=self._settings,
                clock=self._brain._clock,
                margin_provider=None,
                macro_state=self._brain._macro,
            )
            self._brain._generator = SignalGenerator(settings=self._settings)
            self._brain._cooldown = ReentryManager(
                clock=self._brain._clock,
                apply_to_every_exit=self._settings.strategy.cooldown_on_every_exit,
            )
            self._brain._builder = OrderBuilder(
                settings=self._settings,
                clock=self._brain._clock,
                per_trade_risk=self._brain._budget.per_trade_risk,
            )
            self._brain._executor = RoboExecutor(
                client=None,
                builder=self._brain._builder,
                risk=self._brain._risk,
                positions=self._brain._positions,
                journal=self._brain._journal,
                settings=self._settings,
                mode=self._settings.trading_mode,
                clock=self._brain._clock,
            )
            self._brain._reconciler = StateReconciler(
                client=None,
                state_machine=self._brain._state,
                positions=self._brain._positions,
                settings=self._settings,
                journal=self._brain._journal,
                mode=self._settings.trading_mode,
                clock=self._brain._clock,
            )
            self._brain._watchdog = SquareOffWatchdog(
                self._brain._state,
                clock=self._brain._clock,
                drawdown_probe=lambda: self._brain._pnl.total,
                drawdown_limit=self._brain._budget.daily_loss_limit,
                on_drawdown_breach=getattr(self._brain, '_on_drawdown_breach', lambda net: None),
            )
            self._brain._state_publisher = StatePublisher(settings=self._settings, clock=self._brain._clock)
            self._brain._trades = TradeLogger(clock=self._brain._clock)
            self._brain._alerts = TelegramAlerter.from_settings(self._settings, clock=self._brain._clock)
            self._brain._fills = OrderStatusListener(
                on_closed=getattr(self._brain, '_on_fill_closed', lambda *a, **k: None),
                symbol_resolver=watchlist_resolver(
                    {item.token: item.symbol for item in self._settings.watchlist},
                    {item.symbol: item.symbol for item in self._settings.watchlist},
                ),
                on_fill=getattr(self._brain, '_publish_fill', lambda *a, **k: None),
                on_booked_fill=self._brain._trades.record_fill,
                on_rogue_fill=getattr(self._brain, '_on_rogue_fill', lambda *a, **k: None),
                daily_lock=DailyLock(),
                journal=self._brain._journal,
                mode=self._settings.trading_mode,
                clock=self._brain._clock,
            )
            self._brain._poller = OrderBookPoller(
                client=None,
                listener=self._brain._fills,
                clock=self._brain._clock,
                on_degraded=lambda detail: self._brain._state_publisher.publish_risk(
                    "POLLER",
                    "",
                    "ORDER_BOOK_UNREACHABLE",
                    detail,
                    severity="WARNING",
                ),
            )
            self._brain._owns_subscriber = True
            self._brain._subscriber = None
            self._brain._symbols = {}
            self._brain._entries_in_flight = set()
            self._brain._entry_tasks = set()
            self._brain._tasks = []
            self._brain._stopping = asyncio.Event()
            self._brain.stats = BrainStats()

        # Warm up math engine
        if not warmup():
            self._telemetry.log_operation("warmup_failed", level=logging.CRITICAL)
            raise RuntimeError("Math engine warmup failed")

        # Build per-symbol aggregators
        for item in self._settings.watchlist:
            self._brain._symbols[item.symbol] = _SymbolState(
                aggregator=TickAggregator.from_settings(item.symbol, self._settings)
            )

        self._telemetry.log_operation("orchestrator_initialized", paper_mode=True)

    async def _periodic_telemetry(self) -> None:
        """Emit periodic telemetry snapshots."""
        while True:
            try:
                await asyncio.sleep(10)  # Every 10 seconds

                # Account summary
                if self._paper_engine and self._telemetry:
                    summary = self._paper_engine.get_account_summary()
                    self._telemetry.log_account_summary(summary)
                    self._telemetry.log_operation("telemetry_snapshot", **summary)

                # Position updates
                if self._paper_engine and self._telemetry:
                    positions = self._paper_engine.get_positions()
                    for symbol, pos in positions.items():
                        self._telemetry.log_position_update(
                            symbol=symbol,
                            quantity=pos["quantity"],
                            side=pos["side"],
                            entry_price=Decimal(pos["avg_entry_price"]),
                            current_price=Decimal(pos["avg_entry_price"]),  # Would need LTP
                            unrealised_pnl=Decimal(pos["unrealised_pnl"]),
                            stop_loss=Decimal(pos["stop_loss"]),
                            target=Decimal(pos["target"]),
                        )

            except asyncio.CancelledError:
                break
            except Exception as exc:  # noqa: BLE001
                _log.error("orchestrator.telemetry_failed", error=str(exc))

    async def _paper_engine_telemetry(self) -> None:
        """Monitor paper engine and emit trade events."""
        # This would be driven by the paper engine's events
        # For now, just keep the task alive
        while True:
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                break

    async def _teardown(self) -> None:
        """Stop everything in the correct order."""
        _log.info("orchestrator.teardown_started")

        # Stop brain
        if self._brain:
            with contextlib.suppress(TimeoutError, Exception):
                await asyncio.wait_for(self._brain.stop(), timeout=BRAIN_SHUTDOWN_SECONDS)
            with contextlib.suppress(TimeoutError, Exception):
                await asyncio.wait_for(self._brain.shutdown(), timeout=BRAIN_SHUTDOWN_SECONDS)

        # Stop paper engine
        if self._paper_engine:
            with contextlib.suppress(Exception):
                await self._paper_engine.stop()

        # Stop supervisor
        await self._supervisor.stop()

        # Cancel all tasks
        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

        # Close client
        if self._client:
            with contextlib.suppress(Exception):
                await self._client.aclose()

        # Close telemetry
        if self._telemetry:
            with contextlib.suppress(Exception):
                self._telemetry.close()

        # Flush logs
        await flush_all_logs()

        # Close client sessions
        await close_client_sessions()

        # Shutdown logging
        shutdown_logging()

        _log.info("orchestrator.teardown_completed")

    def register_shutdown_phase(
        self,
        name: str,
        coroutine: Callable[[], Any],
        timeout: float = 10.0,
        critical: bool = True,
    ) -> None:
        """Register a shutdown phase with the graceful shutdown handler."""
        if self._shutdown:
            self._shutdown.register_component(name, coroutine, timeout, critical)


async def _main() -> int:
    settings = get_settings()

    # Force paper mode for headless deployment
    if settings.trading_mode == TradingMode.LIVE:
        _log.warning("orchestrator.live_mode_disabled", hint="forcing PAPER mode for headless")
        # We don't change the setting object (it's frozen), but we log the override

    configure_logging(
        role="tachyon",
        level=settings.log_level,
        log_dir=settings.log_dir,
        mode=TradingMode.PAPER,  # Force paper mode
    )
    log = get_logger("tachyon.main")

    if not settings.watchlist:
        log.critical("orchestrator.empty_watchlist", hint="add symbols to config/settings.yaml")
        return 2

    # Create orchestrator with graceful shutdown
    async with GracefulShutdown() as shutdown:
        orchestrator = Orchestrator(settings=settings)

        # Register shutdown phases
        shutdown.register_component("brain", orchestrator.brain.shutdown if orchestrator.brain else lambda: None)
        shutdown.register_component("paper_engine", orchestrator.paper_engine.stop if orchestrator.paper_engine else lambda: None)
        shutdown.register_component("supervisor", orchestrator.supervisor.stop)
        shutdown.register_component("telemetry", orchestrator.telemetry.close if orchestrator.telemetry else lambda: None)

        orchestrator._shutdown = shutdown

        # Notify systemd we're ready
        notify_systemd_ready()

        try:
            return await orchestrator.run()
        except asyncio.CancelledError:
            log.info("orchestrator.cancelled")
            return 0


async def build_client(settings: Settings) -> SmartApiClient | None:
    """Build and log in a SmartAPI client.

    Returns None if credentials are missing (paper mode).
    Raises SmartApiError if login fails in LIVE mode.
    """
    if not settings.smartapi_password.get_secret_value():
        if settings.trading_mode == TradingMode.LIVE:
            raise SmartApiError("login", "LIVE mode with no SMARTAPI_PASSWORD configured")
        return None

    client = SmartApiClient(settings=settings)
    await client.login()
    return client


def confirm_live(settings: Settings) -> bool:
    """Confirm live trading mode.

    Requires the user to type exactly "LIVE" to confirm. This is a safety measure
    to prevent accidental live trading.
    """
    if settings.trading_mode != TradingMode.LIVE:
        return True
    try:
        answer = input(
            "\n" + "=" * 74 + "\n"
            "  TRADING_MODE=LIVE — this session will place REAL ORDERS with REAL MONEY.\n"
            f"  Daily loss limit: Rs.500 (latching)   Watchlist: {len(settings.watchlist)} symbols\n"
            "  Auto square-off at 15:15 IST is unconditional.\n"
            "=" * 74 + "\n"
            "  Type LIVE to confirm, anything else to abort: "
        ).strip()
        return answer == "LIVE"
    except (EOFError, KeyboardInterrupt):
        return False


def main() -> int:
    """Console entry point."""
    try:
        return eventloop.run(_main())
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())