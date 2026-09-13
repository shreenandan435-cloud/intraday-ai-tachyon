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

import argparse
import asyncio
import contextlib
import functools
import logging
import math
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Final

from tachyon.core import eventloop
from tachyon.core.clock import SYSTEM_CLOCK, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.constants import PROJECT_ROOT, TradingMode
from tachyon.core.logger import configure_logging, get_logger, shutdown_logging
from tachyon.core.shutdown import (
    GracefulShutdown,
    close_client_sessions,
    flush_all_logs,
    notify_systemd_ready,
)
from tachyon.execution.api import SmartApiClient, SmartApiError
from tachyon.execution.router import ExecutionRouter
from tachyon.ingestion.angel_adapter import AngelOneWebSocketClient
from tachyon.ingestion.shm_writer import SHMWriter
from tachyon.paper import PaperTradingEngine, create_paper_engine
from tachyon.strategy.brain import StrategyBrain, _SymbolState
from tachyon.strategy.scanner import plan_rotation
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


#: The C++ Engine Sidecar executable.
SIDECAR_EXECUTABLE: Final[Path] = PROJECT_ROOT / "build" / "tachyon_sidecar.exe"

#: Default engine plan for the sidecar — the production 25-feature policy exported by
#: :data:`tachyon.model.export.DEFAULT_LIVE_PLAN`. The toy ``test_engine.plan`` fixture is
#: for the hot-swap integration tests only; booting the engine against it would point the
#: sidecar's inference at a 4-float dummy contract instead of the LOB state vector.
SIDECAR_ENGINE_PLAN: Final[Path] = PROJECT_ROOT / "build" / "tachyon_live.plan"

#: How long the sidecar gets to exit politely before it is killed.
SIDECAR_TERMINATE_GRACE_SECONDS: Final[float] = 10.0

# ── Universe rotation ────────────────────────────────────────────────────────
#: Most one swap per rotation cycle. Every rotation restarts a symbol's VWAP and sequence
#: state, so one deliberate substitution beats three marginal ones — churn is a cost.
ROTATION_MAX_SWAPS: Final[int] = 1

#: A mover must clear this composite score (one cross-sectional standard deviation above the
#: mean mover) before it may displace a watched symbol. Below it, "better than the weakest
#: watched" is not the bar — *meaningfully* better is.
ROTATION_MIN_SCORE: Final[float] = 1.0

#: The rescan selects well beyond the watchlist size so the rotation planner sees scores for
#: every watched symbol that is still a mover, not just the top few.
ROTATION_SCAN_SIZE: Final[int] = 16


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


class SidecarSupervisor:
    """Runs the C++ Engine Sidecar subprocess and keeps it alive.

    Args:
        enabled: False skips launching the sidecar.
    """

    __slots__ = ("_enabled", "_process", "_settings", "_stopping", "stats")

    def __init__(self, *, enabled: bool = True, settings: Settings | None = None) -> None:
        self._enabled = enabled
        self._settings = settings if settings is not None else get_settings()
        self._process: asyncio.subprocess.Process | None = None
        self._stopping = asyncio.Event()
        self.stats: dict[str, Any] = {"starts": 0, "exits": 0, "code": None}

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.returncode is None

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    async def run(self) -> None:
        """Keep the sidecar alive until :meth:`stop`. Mirrors IngestorSupervisor.

        A one-shot run would log a crash and leave inference unavailable for the rest of the
        session; this restarts on a budget with a stability reset instead. stdio is inherited
        (not PIPE) because nothing drains those pipes — an undrained PIPE fills its buffer and
        deadlocks a long-running child.
        """
        if not self._enabled:
            return
        if not SIDECAR_EXECUTABLE.is_file():
            _log.warning(
                "sidecar.missing_executable",
                path=str(SIDECAR_EXECUTABLE),
                impact="sidecar will not start; inference is unavailable",
            )
            return

        restarts = 0
        while not self._stopping.is_set():
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                # Both endpoints are passed explicitly from the settings port map: the
                # sidecar's compiled-in control default once pointed at the tick spine
                # (5555), and relying on it made the sidecar race the Ingestor's PUB bind
                # for the same port. Never let a child default decide the topology.
                self._process = await asyncio.create_subprocess_exec(
                    str(SIDECAR_EXECUTABLE),
                    "--engine",
                    str(SIDECAR_ENGINE_PLAN),
                    "--endpoint",
                    self._settings.zmq_sidecar_control_endpoint,
                    "--actions-endpoint",
                    self._settings.zmq_action_endpoint,
                    cwd=str(PROJECT_ROOT),
                )
                self.stats["starts"] += 1
                _log.info(
                    "sidecar.started",
                    pid=self._process.pid,
                    executable=str(SIDECAR_EXECUTABLE),
                    engine_plan=str(SIDECAR_ENGINE_PLAN),
                    control_endpoint=self._settings.zmq_sidecar_control_endpoint,
                    actions_endpoint=self._settings.zmq_action_endpoint,
                )
                code = await self._process.wait()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - a spawn fault must not kill the supervisor
                code = -1
                _log.error(
                    "sidecar.spawn_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                )

            self.stats["exits"] += 1
            self.stats["code"] = code
            if self._stopping.is_set():
                _log.info("sidecar.exited", code=code, reason="shutdown")
                return

            uptime = loop.time() - started
            if uptime >= STABLE_CHILD_SECONDS:
                restarts = 0
            restarts += 1

            if restarts > MAX_INGESTOR_RESTARTS:
                _log.critical(
                    "sidecar.gave_up",
                    restarts=restarts - 1,
                    impact="inference unavailable for the rest of the session",
                )
                return

            _log.warning(
                "sidecar.restarting",
                exit_code=code,
                attempt=restarts,
                uptime_seconds=round(uptime, 1),
            )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=2.0)

    async def stop(self) -> None:
        """Terminate the sidecar and wait for it, escalating to a kill if it will not go."""
        self._stopping.set()
        process = self._process
        if process is None or process.returncode is not None:
            return

        _log.info("sidecar.stopping", pid=process.pid)
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=SIDECAR_TERMINATE_GRACE_SECONDS)
        except TimeoutError:
            _log.error(
                "sidecar.killing",
                pid=process.pid,
                waited_seconds=SIDECAR_TERMINATE_GRACE_SECONDS,
            )
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            with contextlib.suppress(TimeoutError, Exception):
                await asyncio.wait_for(process.wait(), timeout=5.0)


class UniverseRotationCoordinator:
    """Intraday universe rotation — rescans the movers board on the scheduler's IST slots
    (09:30 / 11:30 / 13:30 by default) and re-points subscription slots without dropping the
    WebSocket.

    Policy (all enforced here, in one place):

    * Only a **flat** watched symbol is ever rotated out — no open position, not cooling,
      nothing in flight. A symbol carrying a position keeps its subscription whatever the
      scan says: those ticks drive the exit path.
    * A watched symbol that is *no longer a mover* — absent from the rescan's selection —
      scores ``-inf`` in the new cross-section, so it is first in line for replacement.
    * An incoming mover must clear :data:`ROTATION_MIN_SCORE` and strictly beat the symbol
      it displaces (:func:`~tachyon.strategy.scanner.plan_rotation`).
    * At most :data:`ROTATION_MAX_SWAPS` substitution per cycle.

    The coordinator owns the live ``{symbol: token}`` map; ``settings.watchlist`` is frozen
    at boot and deliberately not rewritten — rotation is a session-time subscription change,
    not a config change.

    Args:
        settings: resolved config (aggregator construction for incoming symbols).
        brain: the strategy brain whose per-symbol state is rotated. May be ``None`` in
            degraded wiring; rotation then only re-points the feed.
        ws_client: the live feed client whose subscription is rotated. May be ``None``;
            rotation then only re-points the brain.
        scanner: intraday universe scanner (live movers source + dynamic floor).
        scheduler: rotation slot scheduler, armed at construction.
        budget: session budget for the affordability screen.
        max_swaps / min_score: rotation policy; see the module constants.
        clock: injectable for tests.
    """

    __slots__ = (
        "_brain",
        "_budget",
        "_clock",
        "_max_swaps",
        "_min_score",
        "_scanner",
        "_scheduler",
        "_settings",
        "_token_by_symbol",
        "_ws",
        "rotations",
    )

    def __init__(
        self,
        *,
        settings: Settings,
        brain: StrategyBrain | None,
        ws_client: AngelOneWebSocketClient | None,
        scanner: Any,
        scheduler: Any,
        budget: Decimal,
        max_swaps: int = ROTATION_MAX_SWAPS,
        min_score: float = ROTATION_MIN_SCORE,
        clock: Any = SYSTEM_CLOCK,
    ) -> None:
        if budget <= 0:
            raise ValueError(f"budget must be positive to screen rotated symbols, got {budget}")
        if max_swaps < 1:
            raise ValueError(f"max_swaps must be at least 1, got {max_swaps}")
        self._settings = settings
        self._brain = brain
        self._ws = ws_client
        self._scanner = scanner
        self._scheduler = scheduler
        self._budget = budget
        self._max_swaps = max_swaps
        self._min_score = min_score
        self._clock = clock
        self._token_by_symbol: dict[str, str] = {
            item.symbol: item.token for item in settings.watchlist
        }
        self.rotations = 0

    @property
    def token_by_symbol(self) -> dict[str, str]:
        """The live subscription map — a copy, so callers cannot mutate it sideways."""
        return dict(self._token_by_symbol)

    async def run(self, stop: asyncio.Event) -> None:
        """Fire on each scheduler slot until ``stop`` is set. Never raises."""
        while not stop.is_set():
            now = now_ist(self._clock)
            wait_seconds = (self._scheduler.next_fire(now) - now).total_seconds()
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(wait_seconds, 0.2))
                return  # stop was set while waiting
            except TimeoutError:
                pass
            slot = self._scheduler.advance()
            if slot is None:
                continue
            try:
                await self._rotate_once(slot)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - one bad cycle must not kill rotation
                _log.error(
                    "rotation.cycle_failed",
                    slot=str(slot),
                    error=str(exc),
                    error_type=type(exc).__name__,
                    impact="this rescan is skipped; the next slot still fires",
                )

    async def _rotate_once(self, slot: datetime) -> None:
        """One rescan-and-rotate cycle."""
        brain = self._brain
        if brain is not None:
            from tachyon.core.state import TradingState  # deferred: keep module import light

            if brain.state is not TradingState.ACTIVE:
                _log.info(
                    "rotation.skipped_inactive_state",
                    slot=str(slot),
                    state=str(brain.state),
                    note="rotation only re-points entries; outside ACTIVE there is nothing to "
                    "rotate towards",
                )
                return

        result = await self._scanner.scan(budget=self._budget)
        if not result.is_usable:
            _log.info(
                "rotation.no_candidates",
                slot=str(slot),
                considered=result.considered,
                note="the rescan found nothing tradable; the watchlist stands",
            )
            return

        scores = {candidate.symbol: candidate.score for candidate in result.selected}
        watched_scores = {
            symbol: scores.get(symbol, -math.inf) for symbol in self._token_by_symbol
        }
        plan = plan_rotation(
            watched_scores=watched_scores,
            rotatable=self._rotatable_symbols(),
            candidates=tuple(scores.items()),
            min_score=self._min_score,
            max_swaps=self._max_swaps,
        )
        if plan.is_empty:
            _log.info("rotation.no_swaps", slot=str(slot), movers=len(scores))
            return

        by_symbol = {candidate.symbol: candidate for candidate in result.selected}
        for swap in plan.swaps:
            self._apply_swap(swap.out_symbol, by_symbol[swap.in_symbol], slot)

    def _rotatable_symbols(self) -> frozenset[str]:
        """Watched symbols that are flat, not cooling and have nothing in flight."""
        brain = self._brain
        if brain is None:
            return frozenset()
        open_symbols = set(brain._positions.open_symbols())  # noqa: SLF001 - orchestrator wiring
        cooling = set(brain._cooldown.cooling_symbols())  # noqa: SLF001
        in_flight = set(brain._entries_in_flight) | set(brain._exits_in_flight)  # noqa: SLF001
        return frozenset(
            symbol
            for symbol in self._token_by_symbol
            if symbol not in open_symbols and symbol not in cooling and symbol not in in_flight
        )

    def _apply_swap(self, out_symbol: str, candidate: Any, slot: datetime) -> None:
        """Drop ``out_symbol``, subscribe the candidate — feed first, then brain state.

        The feed re-point happens even if the brain update fails: a subscribed-but-unwatched
        symbol costs nothing (the brain drops unknown symbols on sight), while a watched
        symbol with no feed would trip FEED_STALE on every entry it considers.
        """
        item = candidate.to_watchlist_item()
        old_token = self._token_by_symbol.pop(out_symbol, None)
        self._token_by_symbol[item.symbol] = item.token

        if self._ws is not None and old_token is not None:
            self._ws.rotate_subscription(add=(item.token,), remove=(old_token,))

        brain = self._brain
        if brain is not None:
            from tachyon.math_engine.core import TickAggregator  # deferred: numba is heavy

            brain._symbols.pop(out_symbol, None)  # noqa: SLF001 - orchestrator wiring
            brain._symbols[item.symbol] = _SymbolState(  # noqa: SLF001
                aggregator=TickAggregator.from_settings(item.symbol, self._settings)
            )

        self.rotations += 1
        _log.info(
            "rotation.swapped",
            slot=str(slot),
            out=out_symbol,
            out_token=old_token,
            in_symbol=item.symbol,
            in_token=item.token,
            score=round(float(candidate.score), 4),
            note="subscription rotated on the live socket; the watchlist config is unchanged",
        )


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
        "_initialized",
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
        paper_engine: PaperTradingEngine | None = None,
        telemetry: TelemetryManager | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        self._supervisor = supervisor if supervisor is not None else IngestorSupervisor()
        self._brain = brain
        self._paper_engine = paper_engine
        self._telemetry = telemetry
        # Future, not Task: the websocket client runs in an executor (run_in_executor
        # returns a Future), and every Task is a Future anyway.
        self._tasks: list[asyncio.Future[None]] = []
        self._shutdown: GracefulShutdown | None = None
        # Idempotency latch: _main may initialize the orchestrator early (so the
        # ExecutionRouter can share the Brain's risk engine), and run() calls
        # _initialize() again — the second call must be a no-op.
        self._initialized = False

    @property
    def brain(self) -> StrategyBrain | None:
        return self._brain

    @property
    def supervisor(self) -> IngestorSupervisor:
        return self._supervisor

    @property
    def paper_engine(self) -> PaperTradingEngine | None:
        return self._paper_engine

    @property
    def telemetry(self) -> TelemetryManager | None:
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
                    state=self._brain.state if hasattr(self._brain, "state") else "UNKNOWN",
                    hint="daily lock engaged, or the broker holds exposure this process did not "
                    "create — flatten it manually and restart",
                )

        # Start background tasks
        self._tasks = [
            asyncio.create_task(self._supervisor.run(), name="ingestor-supervisor"),
            asyncio.create_task(self._periodic_telemetry(), name="periodic-telemetry"),
        ]

        # Start brain if available (always run for telemetry and watchdog, even in read-only mode)
        # The Brain's run() loop IS this process's market-data ingestion: every tick it
        # consumes is evaluated against open positions via RiskEngine.check_exits (the
        # tick-to-exit contract), with protective exits dispatched as their own tasks so
        # the incoming stream never blocks.
        if self._brain:
            self._tasks.append(asyncio.create_task(self._brain.run(), name="brain-run"))

        # Start paper engine telemetry
        if self._paper_engine:
            self._tasks.append(
                asyncio.create_task(self._paper_engine_telemetry(), name="paper-telemetry")
            )

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
        """Initialize all components in order. Idempotent — safe to call twice."""
        if self._initialized:
            return
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
        from tachyon.core.state import DailyLock, StateMachine
        from tachyon.execution.builder import OrderBuilder
        from tachyon.execution.executor import RoboExecutor
        from tachyon.execution.journal import OrderJournal
        from tachyon.execution.reconciliation import OrderBookPoller, StateReconciler
        from tachyon.ipc.monitor import FeedMonitor
        from tachyon.math_engine.core import TickAggregator
        from tachyon.math_engine.warmup import warmup
        from tachyon.persistence.trade_logger import TradeLogger
        from tachyon.risk.budget import SessionBudget
        from tachyon.risk.engine import RiskEngine
        from tachyon.risk.tracker import PnLTracker, PositionRegistry
        from tachyon.risk.watchdog import SquareOffWatchdog
        from tachyon.sentinel.service import SentinelDaemon
        from tachyon.sentinel.state import MacroState
        from tachyon.strategy.brain import BrainStats
        from tachyon.strategy.cooldown import ReentryManager
        from tachyon.strategy.signals import SignalGenerator
        from tachyon.strategy.telemetry import StatePublisher
        from tachyon.ui.postback import OrderStatusListener, watchlist_resolver
        from tachyon.utils.telegram_alerts import TelegramAlerter

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
            if not hasattr(self._brain, "_clock") or self._brain._clock is None:
                self._brain._clock = SYSTEM_CLOCK

            self._brain._settings = self._settings
            self._brain._state = state_machine
            # brain._journal is an OrderJournal (see StrategyBrain.__init__); assigning a
            # TradeLogger here would raise AttributeError on the first OrderJournal call.
            self._brain._journal = OrderJournal(clock=self._brain._clock)
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
                on_trip=getattr(self._brain, "_on_kill_switch", lambda _e: None),
            )
            self._brain._positions = PositionRegistry()
            self._brain._monitor = FeedMonitor(clock=self._brain._clock)
            self._brain._macro = MacroState(
                risk_off_confidence=self._settings.sentinel.risk_off_confidence,
                blacklist_window=timedelta(minutes=self._settings.sentinel.news_blacklist_minutes),
                clock=self._brain._clock,
            )
            self._brain._sentinel = SentinelDaemon(
                state=self._brain._macro, settings=self._settings, clock=self._brain._clock
            )
            self._brain._risk = RiskEngine(
                self._brain._state,
                self._brain._pnl,
                self._brain._monitor,
                self._brain._positions,
                settings=self._settings,
                clock=self._brain._clock,
                margin_provider=None,
                macro_state=self._brain._macro,
                # Overextension guardrail input: (ltp, session_vwap) from the same
                # aggregators the signal generator reads. No extra I/O on the entry path.
                # getattr: test doubles standing in for the Brain may not carry it, and a
                # missing provider merely disables check 12 rather than killing boot.
                quote_provider=getattr(self._brain, "_quote_view", None),
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
            # Narrowed alias for the lambdas below: mypy does not carry the guard's
            # `is not None` narrowing of self._brain into lambda bodies.
            brain = self._brain
            self._brain._watchdog = SquareOffWatchdog(
                self._brain._state,
                clock=self._brain._clock,
                drawdown_probe=lambda: brain._pnl.total,
                drawdown_limit=self._brain._budget.daily_loss_limit,
                on_drawdown_breach=getattr(self._brain, "_on_drawdown_breach", lambda _net: None),
            )
            self._brain._state_publisher = StatePublisher(
                settings=self._settings, clock=self._brain._clock
            )
            self._brain._trades = TradeLogger(clock=self._brain._clock)
            self._brain._alerts = TelegramAlerter.from_settings(
                self._settings, clock=self._brain._clock
            )
            self._brain._fills = OrderStatusListener(
                on_closed=getattr(self._brain, "_on_fill_closed", lambda *_a, **_k: None),
                symbol_resolver=watchlist_resolver(
                    {item.token: item.symbol for item in self._settings.watchlist},
                    {item.symbol: item.symbol for item in self._settings.watchlist},
                ),
                on_fill=getattr(self._brain, "_publish_fill", lambda *_a, **_k: None),
                on_booked_fill=self._brain._trades.record_fill,
                on_rogue_fill=getattr(self._brain, "_on_rogue_fill", lambda *_a, **_k: None),
                daily_lock=DailyLock(),
                journal=self._brain._journal,
                mode=self._settings.trading_mode,
                clock=self._brain._clock,
            )
            self._brain._poller = OrderBookPoller(
                client=None,
                listener=self._brain._fills,
                clock=self._brain._clock,
                on_degraded=lambda detail: brain._state_publisher.publish_risk(
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
        self._initialized = True

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

        # Logged before the flush so it is actually persisted. Logging is NOT shut down here:
        # the caller (_main's finally) performs the final flush and then shutdown_logging, so
        # that no flush ever runs after the handlers are torn down.
        _log.info("orchestrator.teardown_completed")

        # Flush logs
        await flush_all_logs()

        # Close client sessions
        await close_client_sessions()

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


async def _main(args: argparse.Namespace | None = None) -> int:
    # Handle CLI flags before loading settings so .env/env overrides take effect.
    if args is None:
        args = argparse.Namespace(paper=False, live=False, headless=False, mock=None)
    if args.headless:
        os.environ.setdefault("TACHYON_NON_INTERACTIVE", "1")
    if args.live:
        os.environ["TRADING_MODE"] = "LIVE"
        if args.headless:
            os.environ.setdefault("TACHYON_ASSUME_YES", "1")
    elif args.paper:
        os.environ["TRADING_MODE"] = "PAPER"
        if args.headless:
            os.environ.setdefault("TACHYON_NON_INTERACTIVE", "1")

    # The square-off deadline is 15:15 IST, unconditionally (CLAUDE.md §1.1). The only
    # path that may move it is the explicit --mock flag for offline rehearsals. Without
    # the flag the variable is forced blank — env outranks .env and settings.yaml in the
    # config sources, so a stale value in either cannot re-arm the override by accident.
    if args.mock:
        os.environ["MOCK_SQUAREOFF_TIME"] = args.mock
    else:
        os.environ["MOCK_SQUAREOFF_TIME"] = ""

    settings = get_settings()

    # Force paper mode for headless deployment unless explicitly live.
    if args.headless and not args.live and settings.trading_mode == TradingMode.LIVE:
        _log.warning(
            "orchestrator.live_mode_disabled",
            hint="headless mode forces PAPER; use --live for LIVE trading",
        )

    configure_logging(
        role="tachyon",
        level=settings.log_level,
        log_dir=settings.log_dir,
        mode=settings.trading_mode,
    )
    log = get_logger("tachyon.main")

    if args.mock:
        if settings.is_live:
            log.critical(
                "orchestrator.mock_refused_in_live",
                action="--mock moves the 15:15 square-off deadline; it may never run LIVE",
            )
            return 2
        log.warning(
            "orchestrator.mock_squareoff",
            deadline=args.mock,
            action="square-off deadline moved off 15:15 IST — offline rehearsal only",
        )

    if not settings.watchlist:
        log.critical("orchestrator.empty_watchlist", hint="add symbols to config/settings.yaml")
        return 2

    # The LIVE confirmation gate — headless-safe (see confirm_live's resolution order).
    # Pass assume_yes when headless or --live is explicitly requested. confirm_live may
    # block on input(), so run it in an executor: it must not park the event loop, and
    # confirm_live itself stays synchronous (its signature is imported by tests).
    assume_yes = args.headless or args.live
    confirmed = await asyncio.get_running_loop().run_in_executor(
        None, functools.partial(confirm_live, settings, assume_yes=assume_yes)
    )
    if not confirmed:
        log.critical("live.confirmation_refused", action="aborting before any connection")
        return 2

    # Create orchestrator with graceful shutdown
    async with GracefulShutdown() as shutdown:
        orchestrator = Orchestrator(settings=settings)

        # Register shutdown phases. The brain / paper-engine / telemetry callables are
        # resolved lazily at shutdown time, not registration time: those components are all
        # None until Orchestrator._initialize() runs (inside orchestrator.run() below), so
        # binding orchestrator.brain.shutdown here would silently register a no-op and the
        # coordinator's ordered, timeout-bounded teardown would never reach them.
        shutdown.register_component(
            "brain",
            lambda: orchestrator.brain.shutdown() if orchestrator.brain is not None else None,
            # Match the brain's own shutdown budget; the 10 s default could truncate a slow
            # flatten and, with shutdown() latched idempotent, the tail would never re-run.
            timeout=BRAIN_SHUTDOWN_SECONDS,
        )
        shutdown.register_component(
            "paper_engine",
            lambda: (
                orchestrator.paper_engine.stop() if orchestrator.paper_engine is not None else None
            ),
        )
        shutdown.register_component("supervisor", orchestrator.supervisor.stop)
        shutdown.register_component(
            "telemetry",
            lambda: orchestrator.telemetry.close() if orchestrator.telemetry is not None else None,
        )

        # Spawn and supervise the C++ Engine Sidecar
        sidecar_supervisor = SidecarSupervisor(enabled=True, settings=settings)
        sidecar_task = asyncio.create_task(sidecar_supervisor.run(), name="sidecar-supervisor")

        # Initialize the orchestrator before spawning the router so the router can share
        # the Brain's risk engine. The router's standalone fallback carries a FeedMonitor
        # that nothing on the tick spine ever feeds, so every entry it dispatches dies as
        # FEED_STALE two seconds after boot. Orchestrator.run() calls _initialize() again;
        # the idempotency latch makes that a no-op.
        await orchestrator._initialize()

        # Spawn and supervise the ExecutionRouter
        paper_mode = args.paper or (settings.trading_mode == TradingMode.PAPER)
        brain = orchestrator.brain
        router = ExecutionRouter(
            paper_trade=paper_mode,
            settings=settings,
            risk=brain.risk if brain is not None else None,
        )
        router_task = asyncio.create_task(router.run(), name="execution-router")

        # Spawn and supervise the AngelOneWebSocketClient (if credentials available)
        ws_writer: SHMWriter | None = None
        websocket_client: AngelOneWebSocketClient | None = None
        ws_task: asyncio.Future[None] | None = None
        if settings.smartapi_api_key.get_secret_value() and settings.watchlist:
            try:
                from tachyon.ingestion.angel_adapter import load_feed_secrets

                ws_writer = SHMWriter()
                secrets = load_feed_secrets()
                websocket_client = AngelOneWebSocketClient(
                    writer=ws_writer,
                    tokens=[item.token for item in settings.watchlist],
                    secrets=secrets,
                )
                ws_task = asyncio.get_running_loop().run_in_executor(None, websocket_client.run)
            except Exception as exc:
                log.warning(
                    "main.websocket_client_failed",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    impact="feed will not connect; trading continues against existing SHM data",
                )

        # Universe rotation — hourly rescans (09:30 / 11:30 / 13:30 IST) that swap a flat
        # watched symbol for a newly emerging mover on the live socket. Requires the feed
        # client (a rotated symbol with no ticks would trip FEED_STALE), a brain (rotation
        # targets the watched state) and a dynamic budget (the affordability screen needs a
        # real number; under the §1 constants there is nothing to size against).
        rotation_task: asyncio.Task[None] | None = None
        rotation_stop: asyncio.Event | None = None
        rotation_budget = settings.capital.session_budget_inr
        if (
            websocket_client is not None
            and orchestrator.brain is not None
            and rotation_budget > 0
        ):
            try:
                from tachyon.ingestion.instruments import InstrumentMaster
                from tachyon.strategy.scanner import (
                    AngelMoversSource,
                    DualSourcePreOpen,
                    IntradayScanner,
                    MasterSymbolResolver,
                    NsePreOpenSource,
                    RotationScheduler,
                    cached_angel_headers,
                )

                resolver = MasterSymbolResolver.from_master(InstrumentMaster())
                api_key = settings.smartapi_api_key.get_secret_value()
                client_code = settings.smartapi_client_code.get_secret_value()
                primary_source: Any = NsePreOpenSource()
                # Dual-source only when the credentials to authenticate the fallback exist;
                # the JWT itself is resolved from the token cache at fetch time, not here.
                universe_source = (
                    DualSourcePreOpen(
                        primary_source,
                        AngelMoversSource(
                            header_provider=cached_angel_headers(
                                api_key=api_key, client_id=client_code
                            )
                        ),
                    )
                    if api_key and client_code
                    else primary_source
                )
                coordinator = UniverseRotationCoordinator(
                    settings=settings,
                    brain=orchestrator.brain,
                    ws_client=websocket_client,
                    scanner=IntradayScanner(
                        source=universe_source, resolver=resolver, size=ROTATION_SCAN_SIZE
                    ),
                    scheduler=RotationScheduler(),
                    budget=rotation_budget,
                )
                rotation_stop = asyncio.Event()
                rotation_task = asyncio.create_task(
                    coordinator.run(rotation_stop), name="universe-rotation"
                )
                log.info(
                    "main.rotation_armed",
                    slots=[str(slot) for slot in coordinator._scheduler.times],  # noqa: SLF001
                    max_swaps=ROTATION_MAX_SWAPS,
                    min_score=ROTATION_MIN_SCORE,
                )
            except Exception as exc:  # noqa: BLE001 - rotation is an enhancement, not the feed
                log.warning(
                    "main.rotation_unavailable",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    impact="the pre-open watchlist stands for the whole session",
                )

        # Register shutdown phases for new components
        async def _shutdown_sidecar() -> None:
            await sidecar_supervisor.stop()
            if not sidecar_task.done():
                sidecar_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await sidecar_task

        async def _shutdown_router() -> None:
            router.stop()
            if not router_task.done():
                router_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await router_task

        async def _shutdown_websocket() -> None:
            loop = asyncio.get_running_loop()
            if websocket_client is not None:
                # stop() is the ONLY thing that unblocks the SDK's blocking run loop: it
                # closes the active socket AND raises the stopping flag, so the adapter's
                # outer reconnect loop exits instead of dialling again. The future returned
                # by run_in_executor cannot be cancelled — its non-daemon worker thread
                # keeps running regardless — so we trigger stop(), then give that thread a
                # bounded window to finish instead of pretending cancel() works.
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(None, websocket_client.stop), timeout=5.0
                    )
                except Exception as exc:  # noqa: BLE001 - log it; never swallow silently
                    log.warning(
                        "main.websocket_close_failed",
                        error=str(exc),
                        error_type=type(exc).__name__,
                        impact="the feed thread may not have stopped; it is bounded below",
                    )
            # ORDERING CONTRACT: the WebSocket stream must be fully disconnected (the
            # feed thread joined, or its deadline expired) BEFORE the SHM view is
            # released. Releasing the view first leaves on_data racing a released
            # memoryview — the "operation forbidden on released memoryview object"
            # teardown crash. SHMWriter.write_tick also self-defends against a late
            # straggler tick (counted drop), but the join here is the primary gate.
            if ws_task is not None and not ws_task.done():
                # stop() should let run() return; wait it out with a deadline. Do NOT call
                # ws_task.cancel() — it cannot stop the underlying thread, and an un-joined
                # non-daemon executor thread blocks interpreter exit on a rapid restart.
                with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError, Exception):
                    await asyncio.wait_for(asyncio.shield(ws_task), timeout=5.0)
            if ws_writer is not None:
                ws_writer.close()

        async def _shutdown_rotation() -> None:
            if rotation_stop is not None:
                rotation_stop.set()
            if rotation_task is not None and not rotation_task.done():
                with contextlib.suppress(asyncio.CancelledError, TimeoutError, Exception):
                    await asyncio.wait_for(asyncio.shield(rotation_task), timeout=5.0)

        shutdown.register_component("sidecar", _shutdown_sidecar)
        shutdown.register_component("execution_router", _shutdown_router)
        shutdown.register_component("websocket_client", _shutdown_websocket)
        shutdown.register_component("universe_rotation", _shutdown_rotation)

        orchestrator._shutdown = shutdown

        # Notify systemd we're ready
        notify_systemd_ready()

        # Add new background tasks to the orchestrator's supervision list
        # (they run independently but are supervised by the graceful shutdown)
        orchestrator._tasks.extend([sidecar_task, router_task])
        if ws_task is not None:
            orchestrator._tasks.append(ws_task)
        if rotation_task is not None:
            orchestrator._tasks.append(rotation_task)

        # If dry-run flag is set, schedule an automatic shutdown shortly after start
        if args.dry_run:

            async def _auto_shutdown() -> None:
                await asyncio.sleep(0.5)
                # Trigger graceful shutdown; this will stop all components cleanly
                shutdown.request_shutdown("dry_run")

            asyncio.create_task(_auto_shutdown())

        try:
            return await orchestrator.run()
        except asyncio.CancelledError:
            log.info("orchestrator.cancelled")
            return 0
        finally:
            # Ensure sockets and telemetry are flushed even if shutdown raised. All flushes
            # run BEFORE shutdown_logging so nothing is written after the handlers are torn
            # down; shutdown_logging is the very last step and happens exactly once, here.
            await flush_all_logs()
            await close_client_sessions()
            # Flush telemetry explicitly
            if orchestrator.telemetry is not None:
                with contextlib.suppress(Exception):
                    orchestrator.telemetry.close()
            shutdown_logging()


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


def confirm_live(settings: Settings, *, assume_yes: bool | None = None) -> bool:
    """Confirm live trading mode — headless-safe, and now actually wired into ``_main``.

    Resolution order:

    1. Non-LIVE modes pass unconditionally.
    2. ``assume_yes`` / ``TACHYON_ASSUME_YES=1|true|yes`` confirms without a prompt
       (Task Scheduler, systemd, containers). Logged loudly so an operator can see that
       real money moved without a human at the keyboard.
    3. No interactive stdin (daemon/service context) ⇒ **refuse**, never prompt. This is
       the fix for ``OSError: [Errno 9] Bad file descriptor`` crashing background boots:
       on a closed console ``input()`` raises ``OSError``, which the old handler did not
       catch, and the process died mid-boot with positions possibly open elsewhere.
    4. Otherwise ask, requiring exactly ``LIVE``.
    """
    if settings.trading_mode != TradingMode.LIVE:
        return True

    log = get_logger("tachyon.live_confirm")
    env_flag = os.getenv("TACHYON_ASSUME_YES", "").strip().lower()
    if assume_yes or env_flag in {"1", "true", "yes"}:
        log.warning(
            "live.confirmed_noninteractive",
            source="cli" if assume_yes else "TACHYON_ASSUME_YES",
            impact="this session WILL place real orders with real money",
        )
        return True

    stdin = getattr(sys, "stdin", None)
    if stdin is None or not stdin.isatty():
        log.critical(
            "live.refused_headless",
            reason="no interactive console available",
            hint="run once interactively to confirm, or set TACHYON_ASSUME_YES=1",
            action="refusing to start LIVE trading blind",
        )
        return False
    try:
        answer = input(
            "\n" + "=" * 74 + "\n"
            "  TRADING_MODE=LIVE — this session will place REAL ORDERS with REAL MONEY.\n"
            f"  Daily loss limit: Rs.500 (latching)   "
            f"Watchlist: {len(settings.watchlist)} symbols\n"
            "  Auto square-off at 15:15 IST is unconditional.\n"
            "=" * 74 + "\n"
            "  Type LIVE to confirm, anything else to abort: "
        )
        # No .strip(): "LIVE " with a stray space is a slip, not a confirmation.
        return answer == "LIVE"
    except EOFError, KeyboardInterrupt, OSError:
        # EOFError: piped/closed stdin. OSError errno 9: no console at all (Task Scheduler).
        return False


def main(argv: list[str] | None = None) -> int:
    """Console entry point with CLI flags for headless, paper, and live modes."""
    parser = argparse.ArgumentParser(
        prog="tachyon.main",
        description="Headless master boot sequence for Tachyon",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Force paper trading mode (default for headless)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live trading mode (requires interactive confirmation unless --headless)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Non-interactive headless mode (no prompts, auto-confirm live if --headless)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Boot orchestrator components and shut down immediately for integration testing",
    )
    parser.add_argument(
        "--mock",
        nargs="?",
        const="23:59",
        default=None,
        metavar="HH:MM",
        help="Offline rehearsal only: move the square-off deadline off 15:15 IST "
        "(defaults to 23:59 when no time is given). Refused in LIVE mode; without this "
        "flag the 15:15 IST deadline is unconditional.",
    )
    args = parser.parse_args(argv)
    try:
        return eventloop.run(_main(args=args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
