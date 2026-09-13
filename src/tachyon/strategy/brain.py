"""The Brain — Process B's async orchestrator, CLAUDE.md §2, §4, §6.

Everything built in Phases 2–7 is wired together here, and nothing new is decided. That is the
design: the Brain routes, it does not judge. The math engine computes, the signal generator
proposes, the Risk Engine permits, the executor transmits, the watchdog flattens. If a rule
appears to live in this file, it is in the wrong file.

The loop
--------
::

    async for envelope in subscriber:          # zmq.asyncio — never blocks the loop
        TICK.  → aggregator.on_tick()          # ~1 µs, then maybe evaluate
        DEPTH. → aggregator.on_orderbook()     # refreshes both OBI variants
        FEED.  → heartbeat; liveness only

    every N ticks or on candle close:
        signal   = generator.evaluate(snapshot)         # pure, ~16 µs
        cooldown = reentry.may_enter(symbol)            # §8.1
        decision = risk.evaluate(symbol)                # the only authority (§4)
        plan     = builder.build(...)                   # geometry + sizing (§6)
        report   = await executor.place_robo_order(...) # dispatched as a task

Why the entry is a task, not an ``await``
-----------------------------------------
A bracket placement is two broker round trips — hundreds of milliseconds on a good day,
seconds on a bad one. Awaiting it inline would stop consuming ticks for that whole window, and
the resulting sequence gap would taint the aggregator and void the session VWAP (CLAUDE.md
§3.3). Worse, it would stall the feed-staleness check at exactly the moment we have a live
order. So the entry runs as its own task and the loop keeps draining the socket.

The cost of that choice is that two evaluations could race into a double entry. Three
independent things prevent it: an explicit in-flight set (checked and set synchronously, so
there is no await between test and set), the Risk Engine's ``POSITION_ALREADY_OPEN`` veto, and
the re-entry cooldown. The in-flight set is the one that closes the window the other two cannot
see.

Failure posture
---------------
The loop absorbs every ``Exception`` from message handling and continues. A malformed tick, a
kernel returning something unexpected, a bug we have not found — none of those may take down
the process that owns the square-off watchdog. What is *not* absorbed:
:class:`~tachyon.ipc.subscriber.SchemaVersionMismatchError`, because a wire disagreement means
every price we decode is suspect, and ``BaseException``, because that is the operator asking us
to stop.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final

from tachyon.core.clock import SYSTEM_CLOCK, Clock, now_ist
from tachyon.core.config import Settings, get_settings
from tachyon.core.constants import WATCHDOG_TICK, TradingMode
from tachyon.core.logger import get_logger
from tachyon.core.state import DailyLock, StateMachine, TradingState
from tachyon.core.symbols import normalize_symbol
from tachyon.execution.api import SmartApiClient, SmartApiError
from tachyon.execution.builder import BracketPlan, OrderBuilder, OrderRejected, Side
from tachyon.execution.charges import estimate_charges
from tachyon.execution.executor import RoboExecutor
from tachyon.execution.journal import OrderJournal
from tachyon.execution.reconciliation import OrderBookPoller, StateReconciler
from tachyon.ipc.monitor import FeedMonitor
from tachyon.ipc.schemas import (
    TOPIC_DEPTH,
    TOPIC_HEARTBEAT,
    TOPIC_TICK,
    FillUpdate,
    Heartbeat,
    OrderBook,
    PnLUpdate,
    StateUpdate,
    Tick,
    symbol_from_topic,
)
from tachyon.ipc.subscriber import AsyncSubscriber, Envelope, SubscriberRole
from tachyon.math_engine.core import IndicatorSnapshot, TickAggregator
from tachyon.math_engine.warmup import is_warm, warmup
from tachyon.persistence.trade_logger import TRIGGER_VWAP_CONFLUENCE, TradeLogger
from tachyon.risk.budget import SessionBudget
from tachyon.risk.engine import ExitDecision, RiskDecision, RiskEngine
from tachyon.risk.tracker import PnLTracker, PositionRegistry, TripEvent
from tachyon.risk.watchdog import SquareOffAction, SquareOffWatchdog
from tachyon.sentinel.service import SentinelDaemon
from tachyon.sentinel.state import MacroSnapshot, MacroState
from tachyon.strategy.cooldown import ReentryManager
from tachyon.strategy.signals import Signal, SignalGenerator, SignalReport
from tachyon.strategy.telemetry import StatePublisher, money
from tachyon.ui.postback import OrderStatusListener, OrderUpdate, watchlist_resolver
from tachyon.utils.telegram_alerts import AlertKind, TelegramAlerter

_log = get_logger(__name__)

#: Topics the Brain subscribes to. Heartbeats are included deliberately: without them a
#: consumer cannot tell a quiet market from a dead feed (CLAUDE.md §2.3).
BRAIN_TOPICS: Final[tuple[str, ...]] = (TOPIC_TICK, TOPIC_DEPTH, TOPIC_HEARTBEAT)

#: How often the liveness task runs. Matches the risk watchdog's cadence so the feed-stale
#: verdict and the square-off deadline are evaluated on the same clock granularity.
LIVENESS_INTERVAL_SECONDS: Final[float] = WATCHDOG_TICK.total_seconds()

#: How often free margin is re-read from the broker in LIVE mode. The risk gate's margin check
#: is synchronous by contract (it runs on the entry path and may not await), so it reads a
#: cached value that this task refreshes.
MARGIN_REFRESH_SECONDS: Final[float] = 30.0

ZERO: Final[Decimal] = Decimal("0")
ONE: Final[Decimal] = Decimal("1")


@dataclass(slots=True)
class BrainStats:
    """Counters for observability. Never used for a trading decision."""

    ticks: int = 0
    depth_updates: int = 0
    heartbeats: int = 0
    evaluations: int = 0
    signals: int = 0
    cooldown_blocks: int = 0
    risk_vetoes: int = 0
    build_rejections: int = 0
    entries_placed: int = 0
    entries_failed: int = 0
    handler_errors: int = 0
    unknown_symbols: int = 0


@dataclass(slots=True)
class _SymbolState:
    """Per-symbol working state. One instance per watchlist entry."""

    aggregator: TickAggregator
    ticks_since_eval: int = 0
    last_candle_count: int = 0
    entries_this_session: int = 0
    last_report: SignalReport | None = field(default=None)


class StrategyBrain:
    """Owns the tick loop, the state machine, and the wiring between every other component.

    Args:
        settings: resolved config.
        client: broker client. ``None`` is permitted only in PAPER.
        subscriber: injected for tests; constructed from config when omitted.
        clock: injected for testing.

    Example::

        brain = StrategyBrain(settings=get_settings(), client=client)
        await brain.boot()
        await brain.run()
    """

    __slots__ = (
        "_alerts",
        "_budget",
        "_builder",
        "_clock",
        "_client",
        "_cooldown",
        "_entries_in_flight",
        "_entry_tasks",
        "_executor",
        "_exit_tasks",
        "_exits_in_flight",
        "_generator",
        "_fills",
        "_journal",
        "_latest_ltp",
        "_macro",
        "_margin",
        "_monitor",
        "_owns_subscriber",
        "_pnl",
        "_poller",
        "_positions",
        "_reconciler",
        "_risk",
        "_sentinel",
        "_settings",
        "_shutdown_done",
        "_state",
        "_state_publisher",
        "_stopping",
        "_subscriber",
        "_symbols",
        "_tasks",
        "_trades",
        "_watchdog",
        "stats",
    )

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: SmartApiClient | None = None,
        subscriber: AsyncSubscriber | None = None,
        state_machine: StateMachine | None = None,
        state_publisher: StatePublisher | None = None,
        daily_lock: DailyLock | None = None,
        trade_logger: TradeLogger | None = None,
        alerts: TelegramAlerter | None = None,
        clock: Clock = SYSTEM_CLOCK,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._clock = clock
        self._client = client

        mode = self._settings.trading_mode
        if mode is TradingMode.LIVE and client is None:
            raise ValueError("LIVE mode requires a SmartApiClient — refusing to run blind")

        # ── core state ───────────────────────────────────────────────────────
        # Injectable because it is a *filesystem* dependency shared by three components (the
        # P&L latch, the boot resolver, and the fill listener's rogue-fill lockdown). A test
        # that could only redirect one of them would leave the others writing the real
        # data/journal/daily_lock.txt — and a lockdown in a test would brick the repo.
        lock = daily_lock if daily_lock is not None else DailyLock(clock=clock)
        self._state = (
            state_machine if state_machine is not None else StateMachine.boot(lock, clock=clock)
        )
        self._journal = OrderJournal(clock=clock)
        # Resolved once, at boot. Every downstream limit is derived from this one object, so
        # the daily kill switch, the per-trade sizing and the UI gauge cannot disagree about
        # how much this session may lose. Falls back to the §1 constants when unconfigured.
        self._budget = SessionBudget.resolve(
            capital=self._settings.capital.session_budget_inr,
            drawdown_pct=self._settings.capital.max_daily_drawdown_pct,
            per_trade_pct=self._settings.capital.per_trade_risk_pct,
        )
        self._budget.log_summary()

        self._pnl = PnLTracker(
            self._state,
            daily_lock=lock,
            mode=mode,
            clock=clock,
            limit=self._budget.daily_loss_limit,
            # Every lockdown path funnels through `trip`, so hooking it here — rather than at
            # each call site — is what makes it impossible to add a breach path that forgets
            # to tell the operator. Late-bound via the method: `self._alerts` is constructed
            # further down this same __init__.
            on_trip=self._on_kill_switch,
        )
        self._positions = PositionRegistry()
        self._monitor = FeedMonitor(clock=clock)

        # ── risk ─────────────────────────────────────────────────────────────
        self._margin: Decimal | None = None
        # ── the advisory layer (CLAUDE.md §5) ────────────────────────────────
        # Constructed unconditionally. With no API key the daemon never starts and the state
        # stays permissive — the Sentinel's absence must not change what the system may do.
        self._macro = MacroState(
            risk_off_confidence=self._settings.sentinel.risk_off_confidence,
            blacklist_window=timedelta(minutes=self._settings.sentinel.news_blacklist_minutes),
            clock=clock,
        )
        self._sentinel = SentinelDaemon(state=self._macro, settings=self._settings, clock=clock)

        self._risk = RiskEngine(
            self._state,
            self._pnl,
            self._monitor,
            self._positions,
            settings=self._settings,
            clock=clock,
            # Synchronous by contract: the gate runs on the entry path and may not await.
            # `_margin_loop` refreshes the reading; None until the first successful read, which
            # leaves the check inert rather than vetoing everything before the first poll.
            margin_provider=self._current_margin if mode is TradingMode.LIVE else None,
            macro_state=self._macro,
            # In-memory (ltp, session_vwap) for the overextension guardrail — same snapshot
            # values the signal generator just used, no extra I/O on the entry path.
            quote_provider=self._quote_view,
        )

        # ── strategy ─────────────────────────────────────────────────────────
        self._generator = SignalGenerator(settings=self._settings)
        self._cooldown = ReentryManager(
            clock=clock,
            apply_to_every_exit=self._settings.strategy.cooldown_on_every_exit,
        )

        # ── execution ────────────────────────────────────────────────────────
        self._builder = OrderBuilder(
            settings=self._settings,
            clock=clock,
            per_trade_risk=self._budget.per_trade_risk,
        )
        self._executor = RoboExecutor(
            client=client,
            builder=self._builder,
            risk=self._risk,
            positions=self._positions,
            journal=self._journal,
            settings=self._settings,
            mode=mode,
            clock=clock,
        )
        self._reconciler = StateReconciler(
            client=client,
            state_machine=self._state,
            positions=self._positions,
            settings=self._settings,
            journal=self._journal,
            mode=mode,
            clock=clock,
        )

        # ── the 15:15 deadline ───────────────────────────────────────────────
        # Constructed here but started in boot(), once the event loop exists — the flatten
        # action bridges from the watchdog thread onto the loop and needs a live one.
        # The drawdown hard-stop rides on the watchdog rather than only on the P&L update path.
        # That thread is non-daemon, immortal and monotonic-clocked, so the limit is enforced
        # even if the event loop wedges or the feed dies mid-position (CLAUDE.md §1.1).
        # The deadline is 15:15:00 IST (AUTO_SQUAREOFF_IST), unconditionally: with no
        # override the watchdog arms the §1 constant and nothing else. The only deviation
        # is settings.mock_squareoff_time, which main.py populates solely behind the
        # explicit --mock flag for offline rehearsals — normal execution never sets it,
        # and a blank or malformed value keeps 15:15 in force rather than moving it.
        _override_time = None
        _raw_override = (getattr(self._settings, "mock_squareoff_time", None) or "").strip()
        if _raw_override:
            try:
                _override_time = datetime.strptime(_raw_override, "%H:%M").time()
            except ValueError:
                _log.warning(
                    "brain.squareoff_override_invalid",
                    value=_raw_override,
                    action="ignored — the 15:15 IST deadline stands",
                )
            else:
                _log.critical(
                    "brain.squareoff_override_armed",
                    deadline=_override_time.strftime("%H:%M:%S"),
                    action="MOCK MODE — square-off deadline moved off 15:15 IST; "
                    "never valid against a live broker",
                )
        self._watchdog = SquareOffWatchdog(
            self._state,
            clock=clock,
            drawdown_probe=lambda: self._pnl.total,
            drawdown_limit=self._budget.daily_loss_limit,
            on_drawdown_breach=self._on_drawdown_breach,
            squareoff_override_time=_override_time,
        )

        # ── telemetry + fills (Phase 10) ─────────────────────────────────────
        self._state_publisher = (
            state_publisher
            if state_publisher is not None
            else StatePublisher(settings=self._settings, clock=clock)
        )
        # Injectable for the same reason as `daily_lock`: it owns a filesystem path, and a test
        # that could not redirect it would write into the operator's real data/trades/.
        self._trades = trade_logger if trade_logger is not None else TradeLogger(clock=clock)
        # Advisory, outbound-only operator alerts (§5.1's asymmetry, applied to a notifier).
        # Injectable for the same reason as `trade_logger`: it owns a network dependency, and
        # a test that could not redirect it would post into the operator's real Telegram chat.
        # Inert without TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env.
        self._alerts = (
            alerts
            if alerts is not None
            else TelegramAlerter.from_settings(self._settings, clock=clock)
        )
        self._fills = OrderStatusListener(
            on_closed=self._on_fill_closed,
            symbol_resolver=watchlist_resolver(
                {item.token: item.symbol for item in self._settings.watchlist},
                {item.symbol: item.symbol for item in self._settings.watchlist},
            ),
            on_fill=self._publish_fill,
            on_booked_fill=self._trades.record_fill,
            on_rogue_fill=self._on_rogue_fill,
            daily_lock=lock,
            journal=self._journal,
            mode=mode,
            clock=clock,
        )

        # The slow half of the fill loop. The webhook is fast and silently lossy; this sweeps
        # up whatever it dropped. Idempotency in the listener is what makes re-offering free.
        self._poller = OrderBookPoller(
            client=client,
            listener=self._fills,
            clock=clock,
            on_degraded=lambda detail: self._state_publisher.publish_risk(
                "POLLER",
                "",
                "ORDER_BOOK_UNREACHABLE",
                detail,
                severity="WARNING",
            ),
        )

        self._owns_subscriber = subscriber is None
        self._subscriber = subscriber
        self._symbols: dict[str, _SymbolState] = {}
        self._entries_in_flight: set[str] = set()
        self._entry_tasks: set[asyncio.Task[None]] = set()
        # ── tick-to-exit state (the missed-558.70 fix) ────────────────────────
        #: Latest tick LTP per canonical symbol. Written every tick; read by the per-tick
        #: exit guard and by mark-to-market. Decimal(str(float)) keeps prices exact at the
        #: risk boundary (CLAUDE.md §8).
        self._latest_ltp: dict[str, Decimal] = {}
        #: Symbols with a protective exit already dispatched. Checked and set synchronously
        #: in the tick handler, so two ticks can never both launch an exit for one symbol.
        self._exits_in_flight: set[str] = set()
        self._exit_tasks: set[asyncio.Task[None]] = set()
        self._tasks: list[asyncio.Task[None]] = []
        self._stopping = asyncio.Event()
        #: Latch making :meth:`shutdown` idempotent. The coordinator phase, this class's own
        #: ``run()`` finally, and the orchestrator's teardown may each invoke shutdown; only
        #: the first may run the teardown, the rest must be no-ops (no double close of the
        #: subscriber/trades/alerts).
        self._shutdown_done = False
        self.stats = BrainStats()

    # ── inspection ───────────────────────────────────────────────────────────

    @property
    def state(self) -> TradingState:
        return self._state.state

    @property
    def risk(self) -> RiskEngine:
        return self._risk

    @property
    def pnl(self) -> PnLTracker:
        return self._pnl

    @property
    def positions(self) -> PositionRegistry:
        return self._positions

    @property
    def cooldown(self) -> ReentryManager:
        return self._cooldown

    @property
    def executor(self) -> RoboExecutor:
        return self._executor

    @property
    def feed_monitor(self) -> FeedMonitor:
        return self._monitor

    @property
    def macro(self) -> MacroSnapshot:
        """Current Sentinel view. Advisory — the gate reads it, nothing else acts on it."""
        return self._macro.snapshot()

    def aggregator(self, symbol: str) -> TickAggregator | None:
        state = self._symbols.get(symbol)
        return None if state is None else state.aggregator

    def snapshot(self, symbol: str) -> IndicatorSnapshot | None:
        state = self._symbols.get(symbol)
        return None if state is None else state.aggregator.snapshot()

    def _current_margin(self) -> Decimal:
        """Latest cached free-cash reading for the risk gate's margin check.

        Returns ``Decimal('Infinity')`` before the first successful poll, which the check reads
        as undefined and turns into a veto — not into permission. A margin check that passes
        because nothing has been read yet would be worse than not having one.
        """
        return self._margin if self._margin is not None else Decimal("Infinity")

    # ── boot ─────────────────────────────────────────────────────────────────

    async def boot(self) -> bool:
        """Prepare everything that must be true before a tick may be processed.

        Order is load-bearing:

        1. **Warm the JIT kernels.** Cold-JIT on a live tick is a bug (CLAUDE.md §3), and
           ``warmup()`` also self-tests each kernel — a process that computes wrong numbers
           must not reach the market.
        2. **Reconcile against the broker.** Never assume flat (CLAUDE.md §9). A mismatch, or
           a broker we cannot reach, locks the session here rather than after the first entry.
        3. **Build the per-symbol aggregators**, which refuse to construct if step 1 failed.
        4. **Arm the square-off watchdog**, last, so it is running before any tick can arrive.

        Returns:
            True if the process may trade. False means it is locked and will run read-only —
            telemetry still flows, the watchdog still fires, no order can be placed.
        """
        if not is_warm() and not warmup():
            _log.critical(
                "brain.warmup_failed",
                action="LOCKING — kernels failed self-test; refusing to trade on wrong numbers",
            )
            self._lock("MATH_ENGINE_SELF_TEST_FAILED")
            return False

        report = await self._reconciler.run_at_boot()

        for item in self._settings.watchlist:
            self._symbols[item.symbol] = _SymbolState(
                aggregator=TickAggregator.from_settings(item.symbol, self._settings)
            )

        loop = asyncio.get_running_loop()
        self._watchdog.set_square_off_action(
            self._alerting_square_off(
                self._executor.square_off_action(
                    loop,
                    # Live LTPs at attempt time: 15:15 MARKET exits book against the last
                    # seen price instead of deferring (or worse, ₹0.00) when no fill has
                    # landed yet.
                    ltp_provider=self._latest_ltp_snapshot,
                )
            )
        )
        self._watchdog.start()

        tradeable = report.may_trade and not self._state.is_terminal()
        _log.info(
            "brain.booted",
            state=self._state.state,
            mode=self._settings.trading_mode,
            symbols=sorted(self._symbols),
            reconciliation=report.outcome,
            tradeable=tradeable,
        )
        return tradeable

    def _alert(self, send: Callable[[], object]) -> None:
        """Fire one operator alert, absorbing anything it does (CLAUDE.md §5.1).

        :class:`TelegramAlerter` already swallows every delivery failure, so this guards the
        *other* half: building the message. A Decimal that cannot be formatted, or a future
        edit to one of the event helpers, must not raise on the path that books P&L, places an
        entry, or flattens at 15:15. Taking a callable rather than a value is what puts the
        f-string inside the guard instead of at the call site.

        An advisor that can break its caller is not an advisor; it is an undeclared dependency.
        """
        try:
            send()
        except Exception as exc:  # noqa: BLE001 - an alert must never break the trading path
            _log.warning("brain.alert_failed", error=str(exc), error_type=type(exc).__name__)

    def _alerting_square_off(self, inner: SquareOffAction) -> SquareOffAction:
        """Wrap the 15:15 flatten so the operator's phone learns about it (CLAUDE.md §1.1).

        Three constraints shape this, and getting any of them wrong is worse than sending no
        alert at all:

        **It must not change the contract.** The watchdog retries until the account is
        confirmed flat, and it knows to retry because the action *raises*. The exception is
        re-raised untouched; the alert is a side effect on the way past.

        **It must not block.** This runs on the non-daemon watchdog thread, which is the one
        component that still works when everything else has failed. A five-second socket
        timeout here would delay the next flatten attempt by five seconds, so every call is
        an enqueue onto :class:`TelegramAlerter`'s queue.

        **It must not spam.** The watchdog re-attempts on an interval, so a broker that is
        down at 15:15 would otherwise send one alert per retry until 15:30. Each of the three
        messages — fired, first failure, confirmed flat — is sent at most once per session.
        """
        fired = False
        failed = False

        def action() -> None:
            nonlocal fired, failed
            if not fired:
                fired = True
                self._alert(
                    lambda: self._alerts.square_off_started(
                        open_positions=self._positions.open_count
                    )
                )
            try:
                inner()
            except Exception as exc:
                # Bound to a local *before* the lambda: `except ... as exc` unbinds `exc` at
                # the end of the block, so a lambda closing over it would raise NameError
                # when the alerter got round to it.
                detail = f"{type(exc).__name__}: {exc}"
                if not failed:
                    failed = True
                    self._alert(lambda: self._alerts.square_off_failed(detail=detail))
                raise  # the watchdog retries *because* this propagates — never swallow it
            self._alert(lambda: self._alerts.square_off_completed(session_total=self._pnl.total))

        return action

    def _lock(self, reason: str) -> None:
        """Force LOCKED, tolerating a machine that is already terminal.

        The lockdown paths that do **not** run through ``PnLTracker.trip``: an entry whose
        outcome is unknown, an entry that raised, a math kernel failing its self-test. They
        halt the session just as hard as a breach does, and until the alert below they did it
        in silence — an unknown entry outcome can mean a leg is live at the broker with no
        local record of it, which is not something to discover the next morning.
        """
        already_terminal = self._state.is_terminal()
        try:
            self._state.lock_out(reason=reason)
        except Exception as exc:  # noqa: BLE001 - a lockdown that can fail is not a lockdown
            _log.critical("brain.lock_failed", reason=reason, error=str(exc))

        # Only on the transition. A second `_lock` on an already-terminal machine is
        # bookkeeping, and `trip` sends its own message — this must not double up on it.
        if not already_terminal:
            self._alert(lambda: self._alerts.session_locked(reason=reason))

    # ── the loop ─────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Consume the tick spine until :meth:`stop`.

        Starts the liveness and margin tasks alongside, and always tears everything down —
        including the non-daemon watchdog thread, which would otherwise keep the process alive.
        """
        subscriber = self._ensure_subscriber()
        self._tasks = [
            asyncio.create_task(self._liveness_loop(), name="brain-liveness"),
            asyncio.create_task(self._margin_loop(), name="brain-margin"),
            asyncio.create_task(self._sentinel.run(), name="brain-sentinel"),
            asyncio.create_task(self._poller.run(), name="brain-order-poller"),
        ]

        _log.info("brain.running", endpoint=subscriber.endpoint, state=self._state.state)
        try:
            async for envelope in subscriber:
                if self._stopping.is_set():
                    break
                self.on_message(envelope)
        finally:
            await self.shutdown()

    def _ensure_subscriber(self) -> AsyncSubscriber:
        if self._subscriber is None:
            self._subscriber = AsyncSubscriber(
                BRAIN_TOPICS,
                role=SubscriberRole.STRATEGY,
                settings=self._settings,
                monitor=self._monitor,
                clock=self._clock,
            )
        return self._subscriber

    def on_message(self, envelope: Envelope) -> None:
        """Route one decoded message. Never raises.

        Synchronous on purpose: routing plus indicator maths is tens of microseconds, and an
        ``await`` here would let a second message interleave halfway through a symbol's state
        update. Anything genuinely slow — a broker call — is dispatched as a task.
        """
        try:
            message = envelope.message
            if isinstance(message, Tick):
                self._on_tick(symbol_from_topic(envelope.topic), message)
            elif isinstance(message, OrderBook):
                self._on_depth(symbol_from_topic(envelope.topic), message)
            elif isinstance(message, Heartbeat):
                self.stats.heartbeats += 1
        except Exception as exc:  # noqa: BLE001 - one bad message must not stop the loop
            self.stats.handler_errors += 1
            _log.error(
                "brain.handler_failed",
                topic=envelope.topic,
                error=str(exc),
                error_type=type(exc).__name__,
                exc_info=True,
            )

    def _on_tick(self, symbol: str, tick: Tick) -> None:
        state = self._symbols.get(symbol)
        if state is None:
            # Trading a symbol absent from the watchlist is forbidden (CLAUDE.md §8.1), so an
            # unexpected topic is dropped rather than aggregated. The ingestor filters these
            # too; this is the second gate.
            self.stats.unknown_symbols += 1
            return

        self.stats.ticks += 1
        # Record the LTP *before* anything else. This map is what the per-tick exit guard
        # and mark-to-market read; a tick that arrives must instantly be visible to both,
        # whatever else happens downstream.
        canonical = normalize_symbol(symbol)
        self._latest_ltp[canonical] = Decimal(str(tick.ltp))

        state.aggregator.on_tick(tick)
        state.ticks_since_eval += 1

        # Feed the risk gate's VWAP z-score accumulator with the same differenced print the
        # aggregator just consumed. observe_tick is fail-safe and O(1); its verdict is what
        # check 12 consults before authorising a LONG.
        self._risk.observe_tick(canonical, tick.ltp, state.aggregator.last_volume_delta)

        # ── tick-to-exit (the missed-558.70 / rode-past-the-stop fix) ─────────
        # Every single tick is evaluated against open positions. check_exits is pure and
        # allocation-light; with no positions open it costs one integer comparison.
        self._guard_open_positions()

        candles = state.aggregator.candle_count
        candle_closed = candles != state.last_candle_count
        state.last_candle_count = candles

        if (
            candle_closed
            or state.ticks_since_eval >= self._settings.strategy.evaluate_every_n_ticks
        ):
            state.ticks_since_eval = 0
            self._evaluate(symbol, state)

    def _quote_view(self, symbol: str) -> tuple[float, float] | None:
        """Latest ``(ltp, session_vwap)`` for the risk gate's overextension check.

        Synchronous and I/O-free by contract: it reads aggregator state the tick loop just
        wrote. ``None`` when the symbol is unknown to this Brain.

        Reads the two O(1) running accumulators directly. ``snapshot()`` would recompute
        every indicator — five JIT passes over up to 2000 ticks — only to discard all but
        these two values, and this provider is consulted on the entry path.
        """
        state = self._symbols.get(normalize_symbol(symbol))
        if state is None:
            return None
        aggregator = state.aggregator
        if aggregator.session_vwap_valid:
            return (aggregator.ltp, aggregator.session_vwap)
        return None

    def _latest_ltp_snapshot(self) -> dict[str, Decimal]:
        """Point-in-time copy of the live LTP map, for square-off booking."""
        return dict(self._latest_ltp)

    def _guard_open_positions(self) -> None:
        """Evaluate every open position against the latest ticks. Never raises.

        Runs on **every** tick — that is the entire point. The 558.70-target incident and
        the SHORT-ridden-past-its-stop incident both happened because SL/TP comparisons ran
        on a slow strategy cadence or not at all. Here they run at tick rate; only the
        resulting order placement leaves the loop, as its own task.
        """
        if self._positions.open_count == 0:
            return
        try:
            decisions = self._risk.check_exits(self._latest_ltp)
        except Exception as exc:  # noqa: BLE001 - the tick loop must survive a guard bug
            _log.critical(
                "brain.exit_guard_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                action="absorbed; next tick re-evaluates",
                exc_info=True,
            )
            return
        for decision in decisions:
            self._dispatch_exit(decision)

    def _dispatch_exit(self, decision: ExitDecision) -> None:
        """Launch one protective exit as its own task, exactly once per symbol.

        The in-flight marker is checked and set synchronously — no await between test and
        set — so two consecutive ticks cannot both dispatch an exit for the same position
        and turn one flatten into a flatten-plus-reversal. The task itself owns the marker
        afterwards; see :meth:`_execute_protective_exit`.
        """
        symbol = normalize_symbol(decision.symbol)
        if symbol in self._exits_in_flight:
            return
        self._exits_in_flight.add(symbol)
        _log.warning(
            "brain.exit_dispatched",
            symbol=symbol,
            reason=decision.reason.value,
            direction=decision.direction,
            quantity=decision.quantity,
            ltp=str(decision.ltp),
            threshold=str(decision.threshold),
        )
        task = asyncio.create_task(
            self._execute_protective_exit(decision), name=f"brain-exit-{symbol}"
        )
        self._exit_tasks.add(task)
        task.add_done_callback(self._exit_tasks.discard)

    async def _execute_protective_exit(self, decision: ExitDecision) -> None:
        """Transmit a protective exit off the tick path, then book per mode. Never raises.

        PAPER books here and now — there is no broker fill stream coming — through the same
        :meth:`on_position_closed` authority the fill listener uses, so cooldowns, the
        registry and realised P&L move exactly once.

        LIVE registers the exit order with the fill listener and lets the reconciler book
        against the real fill price. Booking an LTP estimate here would double with that
        fill seconds later.
        """
        symbol = normalize_symbol(decision.symbol)
        was_stop_out = decision.reason.value == "STOP_LOSS_HIT"
        quantity = abs(decision.quantity)
        try:
            report = await self._executor.execute_exit(decision)
        except Exception as exc:  # noqa: BLE001 - an exit failure must not kill the tick loop
            _log.critical(
                "brain.exit_raised",
                symbol=symbol,
                reason=decision.reason.value,
                error=str(exc),
                error_type=type(exc).__name__,
                action="position remains open — next tick retries unless already attempted",
                exc_info=True,
            )
            # Bound to a local before the lambda: `except ... as exc` unbinds `exc` at the
            # end of the block, so a plain closure would raise NameError when the alerter
            # got round to it.
            detail = f"protective exit {symbol} raised {type(exc).__name__}: {exc}"
            self._alert(lambda: self._alerts.square_off_failed(detail=detail))
            if not self._positions.is_open(symbol):
                self._exits_in_flight.discard(symbol)
            return

        if report.accepted:
            if report.simulated:
                self._book_paper_exit(decision, report.quantity, was_stop_out)
            else:
                # LIVE: the fill that lands will flow through the postback/poller into
                # on_position_closed. Registering the order is what makes its fill ours.
                self._fills.register_order(report.order_id, report.order_tag)
                self._trades.note_order_placed(
                    report.order_id, trigger_reason=f"PROTECTIVE_{decision.reason.value}"
                )
                kind = AlertKind.STOP_LOSS if was_stop_out else AlertKind.TARGET
                self._alert(
                    lambda: self._alerts.send(
                        f"<b>{symbol}</b> {decision.reason.value} exit submitted x{quantity}\n"
                        f"ltp Rs.{decision.ltp:,.2f}  threshold Rs.{decision.threshold:,.2f} "
                        f"(order {report.order_id or 'pending'})",
                        kind,
                    ),
                )

        # Marker policy. Submitted/suppressed: keep holding while the position still shows
        # open, so later ticks do not re-dispatch into a possible reversal; the registry
        # freeing (via fill closure or square-off) releases it. Rejected: release now, so
        # the next tick can retry against a possibly recovered broker.
        if report.retriable or not self._positions.is_open(symbol):
            self._exits_in_flight.discard(symbol)

    def _book_paper_exit(
        self, decision: ExitDecision, quantity: int, was_stop_out: bool
    ) -> None:
        """Book a completed PAPER protective exit through the single close authority."""
        symbol = normalize_symbol(decision.symbol)
        record = self._positions.get(symbol)
        entry_price = record.entry_price if record is not None else None
        direction_sign = ONE if decision.direction == "LONG" else -ONE
        gross = (
            (decision.ltp - entry_price) * abs(quantity) * direction_sign
            if entry_price is not None
            else ZERO
        )
        buy_turnover = abs(quantity) * (entry_price if entry_price is not None else decision.ltp)
        sell_turnover = abs(quantity) * decision.ltp
        if decision.direction == "SHORT":
            buy_turnover, sell_turnover = sell_turnover, buy_turnover
        charges = estimate_charges(
            buy_turnover=buy_turnover, sell_turnover=sell_turnover, orders=2
        ).total
        # Order matters, as in on_position_closed: cooldown first, then free the slot, then
        # book — no instant where the symbol looks flat and cooldown-free.
        self.on_position_closed(
            symbol,
            realised_inr=gross,
            charges_inr=charges,
            was_stop_out=was_stop_out,
        )
        self._trades.record_close(symbol, gross, charges, was_stop_out)
        self._state_publisher.publish_risk(
            "EXIT",
            symbol,
            decision.reason.value,
            f"ltp={decision.ltp} threshold={decision.threshold} "
            f"realised={gross} charges={charges}",
            severity="WARNING",
        )
        self._alert(
            lambda: self._alerts.position_closed(
                symbol=symbol,
                realised=gross,
                charges=charges,
                was_stop_out=was_stop_out,
                session_total=self._pnl.total,
                headroom=self._pnl.headroom,
            )
        )

    def _on_depth(self, symbol: str, book: OrderBook) -> None:
        state = self._symbols.get(symbol)
        if state is None:
            self.stats.unknown_symbols += 1
            return
        self.stats.depth_updates += 1
        state.aggregator.on_orderbook(book)

    # ── the decision chain ───────────────────────────────────────────────────

    def _evaluate(self, symbol: str, state: _SymbolState) -> None:
        """Steps C–F of the loop. Synchronous up to the point of transmission.

        Ordered cheapest-first, exactly like the risk gate: a symbol in cooldown must not cost
        a snapshot, and a NEUTRAL signal must not cost a risk evaluation.
        """
        self.stats.evaluations += 1
        report = self._generator.evaluate(state.aggregator.snapshot())
        state.last_report = report

        if not report.signal.is_directional:
            return
        self.stats.signals += 1

        # An entry already on the wire for this symbol. Checked and set with no await in
        # between, so two evaluations cannot both pass.
        if symbol in self._entries_in_flight:
            _log.info("brain.entry_in_flight", symbol=symbol, signal=report.signal)
            return

        cap = self._settings.strategy.max_signals_per_symbol_per_session
        if cap and state.entries_this_session >= cap:
            _log.info(
                "brain.session_cap_reached",
                symbol=symbol,
                entries=state.entries_this_session,
                cap=cap,
            )
            return

        verdict = self._cooldown.check(symbol, now_ist(self._clock))
        if not verdict.allowed:
            self.stats.cooldown_blocks += 1
            _log.info("brain.cooldown_block", symbol=symbol, detail=verdict.detail)
            return

        decision = self._risk.evaluate(
            symbol,
            direction="LONG" if report.signal is Signal.LONG else "SHORT",
            # The exact (ltp, session_vwap) this signal was derived from — the guardrail
            # judges those numbers, not a re-read that could have drifted.
            quote=(report.ltp, report.session_vwap),
        )
        if not decision.allowed:
            self.stats.risk_vetoes += 1
            # Already logged with its reason by the gate itself; persisted for the post-mortem,
            # where "why did it not trade" is the question the CSV has to answer.
            self._trades.record_veto(decision)
            return

        try:
            plan = self._builder.build(
                symbol=symbol,
                side=Side.BUY if report.signal is Signal.LONG else Side.SELL,
                entry_price=report.ltp,
                atr=report.atr_5m,
                headroom=self._pnl.headroom,
                # The Sentinel's only quantitative influence. Derived in our code, clamped
                # to [0, 1], and monotonically non-increasing for the session (§5).
                sentinel_multiplier=self._macro.size_multiplier,
            )
        except OrderRejected as exc:
            self.stats.build_rejections += 1
            self._journal.decision(
                "entry_not_built", symbol=symbol, reason=exc.reason, detail=exc.detail
            )
            _log.info("brain.plan_rejected", symbol=symbol, reason=exc.reason, detail=exc.detail)
            return

        self._entries_in_flight.add(symbol)
        task = asyncio.create_task(
            self._place(symbol, state, plan, decision), name=f"brain-entry-{symbol}"
        )
        # Held in a set with a done-callback rather than appended to a list: an all-day session
        # would otherwise accumulate one Task object per entry, and — worse — a task whose
        # exception is never retrieved logs a warning at interpreter shutdown.
        self._entry_tasks.add(task)
        task.add_done_callback(self._entry_tasks.discard)

    async def _place(
        self,
        symbol: str,
        state: _SymbolState,
        plan: BracketPlan,
        decision: RiskDecision,
    ) -> None:
        """Transmit a bracket off the tick path. Never raises.

        The in-flight flag is released in ``finally`` — including on an unknown outcome, where
        it is released *after* the session has been locked, so nothing can slip through the
        gap.
        """
        try:
            report = await self._executor.place_robo_order(plan, decision)
        except Exception as exc:  # noqa: BLE001 - an entry must not take down the Brain
            self.stats.entries_failed += 1
            _log.critical(
                "brain.entry_raised",
                symbol=symbol,
                error=str(exc),
                error_type=type(exc).__name__,
                action="LOCKING — an entry failed in a way we did not anticipate",
                exc_info=True,
            )
            self._lock(f"ENTRY_FAILED: {type(exc).__name__}")
            self._entries_in_flight.discard(symbol)
            return

        try:
            if report.needs_reconciliation:
                # A leg may be live at the broker with no local record of it. Trading on from
                # here would size the next decision against a position we cannot see.
                _log.critical(
                    "brain.entry_outcome_unknown",
                    symbol=symbol,
                    action="LOCKING — reconcile the order book before restarting",
                )
                self._lock("ENTRY_OUTCOME_UNKNOWN")
            elif report.placed:
                for result in report.results:
                    if result.accepted:
                        self._fills.register_order(result.order_id, result.order_tag)
                        # Starts the latency clock and records why this order exists, so its
                        # fill row can carry both without the fill path knowing either.
                        self._trades.note_order_placed(
                            result.order_id, trigger_reason=TRIGGER_VWAP_CONFLUENCE
                        )
                self.stats.entries_placed += 1
                state.entries_this_session += 1
                _log.info(
                    "brain.entry_placed",
                    symbol=symbol,
                    quantity=report.quantity_placed,
                    simulated=report.simulated,
                )
                # Enqueue-and-return: this is the event loop, and §2.2 bans a blocking call
                # on it. The alert carries the geometry, because "we bought RELIANCE" without
                # the stop is not enough for an operator to judge whether to intervene.
                self._alert(
                    lambda: self._alerts.entry_placed(
                        symbol=symbol,
                        direction="LONG" if plan.side is Side.BUY else "SHORT",
                        quantity=report.quantity_placed,
                        entry=plan.geometry.entry_price,
                        stop=plan.geometry.stop_loss_price,
                        target=plan.geometry.target_1_price,
                        simulated=report.simulated,
                    )
                )
            else:
                self.stats.entries_failed += 1
                _log.warning(
                    "brain.entry_not_placed",
                    symbol=symbol,
                    reason=report.rejected_reason,
                    detail=report.rejected_detail,
                )
        finally:
            self._entries_in_flight.discard(symbol)

    # ── position lifecycle ───────────────────────────────────────────────────

    def on_position_closed(
        self,
        symbol: str,
        *,
        realised_inr: Decimal,
        charges_inr: Decimal = Decimal("0"),
        was_stop_out: bool = False,
    ) -> None:
        """Book a closed trade: P&L, exposure, and the re-entry cooldown.

        Called by the fill handler (Phase 10 wires it to the broker's postback stream) and by
        the square-off path. Order matters — the cooldown starts before the registry frees the
        symbol, so there is no instant in which the symbol is both flat and cooldown-free.
        """
        moment = now_ist(self._clock)
        self._cooldown.record_exit(symbol, was_stop_out=was_stop_out, at=moment)
        self._positions.record_exit(symbol, was_stop_out=was_stop_out, at=moment)
        self._pnl.book_realised(realised_inr, charges_inr)
        _log.info(
            "brain.position_closed",
            symbol=symbol,
            realised_inr=str(realised_inr),
            was_stop_out=was_stop_out,
            total_pnl=str(self._pnl.total),
            headroom=str(self._pnl.headroom),
        )

    def _on_kill_switch(self, event: TripEvent) -> None:
        """Push the §1.2 lockdown to the operator's phone.

        Called from :meth:`PnLTracker.trip` after the latch, the state transition and the
        durable lock are all done, so nothing here can delay the lockdown — and on whichever
        thread tripped it, which is the watchdog's for a drawdown breach and the fill loop's
        for a rogue fill. The alerter is thread-safe and enqueue-only, so both are fine.
        """
        self._alert(
            lambda: self._alerts.kill_switch(
                reason=event.reason,
                detail=event.detail,
                realised=event.realised,
                total=event.total,
                limit=event.limit,
                lock_durable=event.lock_durable,
            )
        )

    def _on_drawdown_breach(self, net_inr: Decimal) -> None:
        """Latch the kill switch from the watchdog thread.

        Routed through :meth:`PnLTracker.trip` rather than transitioning the state machine
        directly, so a drawdown breach engages exactly the same machinery as the §1.2 loss
        limit: the in-memory latch, ``daily_lock.txt``, and ``LOCKED``. The day lock is the
        part that matters — ``SQUARED_OFF`` alone would be cleared by a restart, and restarting
        must not be a way back into the market.
        """
        self._pnl.trip(
            "DAILY_DRAWDOWN_BREACHED",
            f"net={net_inr} limit={self._budget.daily_loss_limit} "
            f"capital={self._budget.capital} source={self._budget.source}",
        )
        self._state_publisher.publish_risk(
            "RISK",
            "",
            "DAILY_DRAWDOWN_BREACHED",
            f"net={net_inr} limit={self._budget.daily_loss_limit}",
            severity="CRITICAL",
        )

    @property
    def budget(self) -> SessionBudget:
        """The resolved capital budget. One object; every limit downstream derives from it."""
        return self._budget

    def on_mark_to_market(self, floating_inr: Decimal) -> None:
        """Update unrealised P&L. May trip the kill switch (CLAUDE.md §1.2)."""
        self._pnl.update_floating(floating_inr)

    # ── fills (Phase 10) ─────────────────────────────────────────────────────

    @property
    def poller(self) -> OrderBookPoller:
        """The order-book sweep. Runs alongside the webhook, never instead of it."""
        return self._poller

    @property
    def fills(self) -> OrderStatusListener:
        """The order-status listener. Fed by the postback webhook and by the order-book poll."""
        return self._fills

    def _on_fill_closed(
        self,
        symbol: str,
        realised: Decimal,
        charges: Decimal,
        was_stop_out: bool,
        at: datetime,  # noqa: ARG002 - fixed callback signature; we re-read the clock ourselves
    ) -> None:
        """Bridge :class:`OrderStatusListener` to :meth:`on_position_closed`.

        This is the wire that was missing until Phase 10: without it a position opened and
        never closed in local state, so P&L stayed at zero, the cooldown never started, and
        ``POSITION_ALREADY_OPEN`` blocked the symbol for the rest of the session.
        """
        self.on_position_closed(
            symbol,
            realised_inr=realised,
            charges_inr=charges,
            was_stop_out=was_stop_out,
        )
        self._trades.record_close(symbol, realised, charges, was_stop_out)
        self._state_publisher.publish_risk(
            "FILL",
            symbol,
            "POSITION_CLOSED",
            f"realised={realised} charges={charges} stop_out={was_stop_out}",
            severity="INFO",
        )
        # Runs on the postback webhook thread or the poller's; enqueue-and-return either way.
        # `was_stop_out` is the listener's verdict from the *order that closed the position*
        # (§7.4), never "did this trade lose money" — the alert must not relabel it.
        self._alert(
            lambda: self._alerts.position_closed(
                symbol=symbol,
                realised=realised,
                charges=charges,
                was_stop_out=was_stop_out,
                session_total=self._pnl.total,
                headroom=self._pnl.headroom,
            )
        )

    def _on_rogue_fill(self, update: OrderUpdate) -> None:
        """A fill arrived for an order this system never placed (CLAUDE.md §9).

        The listener has already written the day lock. This adds the in-memory half — the
        state machine and the P&L latch — so nothing in *this* process can trade on either,
        and publishes the alarm so the operator sees it on the terminal rather than in a log
        file they were not reading.
        """
        _log.critical(
            "brain.rogue_fill",
            symbol=update.symbol,
            order_id=update.order_id,
            action="LOCKED — something is trading this account that is not this process",
        )
        # Before the trip, so the specifics land ahead of the generic kill-switch buzz. Two
        # messages for one event is deliberate here and nowhere else: this is the only event
        # whose correct response is to open the broker terminal immediately.
        self._alert(
            lambda: self._alerts.rogue_fill(
                symbol=update.symbol,
                order_id=update.order_id,
                quantity=update.filled_quantity,
                side=update.side,
                price=update.average_price,
            )
        )
        self._pnl.trip(
            "ROGUE_FILL",
            f"unrecognised fill {update.order_id} on {update.symbol}",
        )
        self._state_publisher.publish_risk(
            "ROGUE_FILL",
            update.symbol,
            "UNRECOGNISED_FILL",
            f"order {update.order_id} was never placed by this system",
            severity="CRITICAL",
        )

    def _publish_fill(self, update: OrderUpdate) -> None:
        """Forward one order-status transition to the UI. Never conflated (§2.1)."""
        self._state_publisher.publish_fill(
            FillUpdate(
                symbol=update.symbol,
                order_id=update.order_id,
                status=update.status,
                side=update.side,
                quantity=update.quantity,
                filled_quantity=update.filled_quantity,
                price=str(update.average_price),
                ts_epoch=now_ist(self._clock).timestamp(),
                order_tag=update.order_tag,
                is_exit=update.is_stop_order,
            )
        )

    # ── telemetry ────────────────────────────────────────────────────────────

    def state_frame(self) -> StateUpdate:
        """The complete session view. Whole-state, not a delta — the UI conflates (§2.1)."""
        macro = self._macro.snapshot()
        return StateUpdate(
            state=str(self._state.state),
            mode=self._settings.trading_mode.value,
            ts_epoch=now_ist(self._clock).timestamp(),
            feed_stale=self._monitor.is_stale,
            feed_age_seconds=round(self._monitor.age, 3),
            open_symbols=tuple(sorted(self._positions.open_symbols())),
            cooling_symbols=tuple(sorted(self._cooldown.cooling_symbols())),
            macro_regime=macro.regime.value,
            macro_confidence=macro.confidence,
            macro_blocks_entries=not macro.is_trading_allowed,
            macro_degraded=macro.degraded,
            daily_lock_engaged=self._pnl.is_breached,
            reason=macro.reason,
        )

    def pnl_frame(self) -> PnLUpdate:
        snapshot = self._pnl.snapshot()
        return PnLUpdate(
            realised=money(snapshot.realised),
            floating=money(snapshot.floating),
            charges=money(snapshot.charges),
            total=money(snapshot.total),
            headroom=money(snapshot.headroom),
            limit=money(snapshot.limit),
            breached=snapshot.breached,
            ts_epoch=now_ist(self._clock).timestamp(),
            trades_today=self.stats.entries_placed,
            capital=money(self._budget.capital),
            drawdown_pct=str(self._budget.drawdown_pct.quantize(Decimal("0.01"))),
            limit_source=self._budget.source,
        )

    def publish_telemetry(self) -> None:
        """Push the current state and P&L. Never raises — the publisher swallows its own."""
        self._state_publisher.publish_state(self.state_frame())
        self._state_publisher.publish_pnl(self.pnl_frame())

    # ── background tasks ─────────────────────────────────────────────────────

    async def _liveness_loop(self) -> None:
        """Drive :meth:`FeedMonitor.check` so staleness is edge-triggered, not polled lazily.

        The gate reads :attr:`FeedMonitor.age` directly and would refuse entries regardless.
        This task exists so the *transition* is logged and any callback fires within one tick
        of the feed dying, rather than whenever the next signal happens to be evaluated.
        """
        while not self._stopping.is_set():
            try:
                self._monitor.check()
                self._update_mark_to_market()
                self._honour_external_lock()
                self.publish_telemetry()
            except Exception as exc:  # noqa: BLE001 - liveness must not die quietly
                _log.error("brain.liveness_failed", error=str(exc), exc_info=True)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=LIVENESS_INTERVAL_SECONDS)

    def _update_mark_to_market(self) -> None:
        """Refresh floating P&L from the latest ticks (CLAUDE.md §1.2).

        Feeds two consumers that must never run on a stale number: the drawdown hard-stop's
        probe (the watchdog thread reads ``self._pnl.total``) and 15:15 square-off booking.
        Positions without a tick yet this session are skipped — an absent price is unknown,
        not zero.
        """
        if self._positions.open_count == 0:
            return
        floating = ZERO
        for symbol, record in self._positions.records_snapshot().items():
            ltp = self._latest_ltp.get(normalize_symbol(symbol))
            if ltp is None or record.entry_price is None or record.quantity <= 0:
                continue
            sign = ONE if record.direction == "LONG" else -ONE
            floating += (ltp - record.entry_price) * Decimal(record.quantity) * sign
        self.on_mark_to_market(floating)

    def _honour_external_lock(self) -> None:
        """Adopt a daily lock written by another process — the UI's panic button (§7.3).

        The risk gate already refuses entries while the lock file is present (check 3), so this
        is not what makes panic work. What it adds is that the *state machine* reflects it, so
        the operator sees LOCKED rather than ACTIVE-but-refusing, and every downstream consumer
        agrees about why nothing is trading.

        One direction only. A lock that disappears never un-locks anything: `LOCKED` is
        absorbing (§4), and a system that can un-halt itself is a system that will.
        """
        if self._state.is_terminal() or not self._pnl.daily_lock.is_engaged():
            return
        _log.critical(
            "brain.external_lock_detected",
            path=str(self._pnl.daily_lock.path),
            action="LOCKING — a daily lock was engaged outside this process",
        )
        self._pnl.trip("EXTERNAL_LOCK", "daily lock engaged by another process")

    async def _margin_loop(self) -> None:
        """Refresh the cached free-cash reading for the risk gate's margin check.

        A failure leaves the previous reading in place and is logged. It does not clear the
        cache to ``None``: a transient broker hiccup should not veto every entry, and a broker
        that is genuinely gone will be caught by the feed-staleness rule and by the fact that
        the order itself would fail.
        """
        if self._client is None or self._settings.trading_mode is not TradingMode.LIVE:
            return

        while not self._stopping.is_set():
            try:
                self._margin = await self._client.available_margin()
            except SmartApiError as exc:
                _log.warning(
                    "brain.margin_refresh_failed",
                    error=str(exc),
                    cached=str(self._margin),
                )
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=MARGIN_REFRESH_SECONDS)

    # ── shutdown ─────────────────────────────────────────────────────────────

    async def stop(self) -> None:
        """Ask the loop to exit. Safe from a signal handler."""
        self._stopping.set()

    async def shutdown(self) -> None:
        """Tear everything down in the order that leaves nothing running.

        The watchdog is stopped **last**: it is a non-daemon thread whose entire purpose is to
        flatten during shutdown (CLAUDE.md §1.1), so it must outlive the components it drives.

        Idempotent: the shutdown phase, ``run()``'s own finally and the orchestrator teardown
        can each reach this method. Only the first invocation runs the teardown; the check and
        the latch are set with no await between them, so two coroutines cannot both enter.
        """
        if self._shutdown_done:
            return
        self._shutdown_done = True
        self._stopping.set()

        # In-flight entries are awaited, not cancelled. Cancelling a coroutine that is halfway
        # through a placement produces exactly the state CLAUDE.md §6.4 exists to prevent: an
        # order that may or may not be live, with nothing recorded either way. The same logic
        # applies to in-flight protective exits — arguably more.
        pending_lifecycles = tuple(self._entry_tasks) + tuple(self._exit_tasks)
        if pending_lifecycles:
            _log.warning("brain.awaiting_orders", count=len(pending_lifecycles))
            await asyncio.gather(*pending_lifecycles, return_exceptions=True)

        # Asked to stop before its task is cancelled, so it closes its HTTP client cleanly
        # rather than leaving a socket for the loop to complain about at teardown.
        await self._sentinel.stop()
        await self._poller.stop()
        self._state_publisher.close()

        for task in self._tasks:
            task.cancel()
        for task in self._tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task
        self._tasks.clear()

        if self._subscriber is not None and self._owns_subscriber:
            self._subscriber.close()

        self._watchdog.stop()

        # Last, and after the watchdog: the summary should describe the session including
        # anything the 15:15 flatten booked on its way out.
        self._trades.close()
        # Also after the watchdog, for the same reason — the square-off alerts are queued from
        # that thread, and closing the sender first would discard them.
        self._alerts.close()

        _log.info(
            "brain.stopped",
            state=self._state.state,
            ticks=self.stats.ticks,
            signals=self.stats.signals,
            entries_placed=self.stats.entries_placed,
            risk_vetoes=self.stats.risk_vetoes,
            cooldown_blocks=self.stats.cooldown_blocks,
            handler_errors=self.stats.handler_errors,
            total_pnl=str(self._pnl.total),
        )

    def reset_session(self) -> None:
        """Clear per-day state at 09:15 IST. Does not clear an engaged daily lock."""
        for state in self._symbols.values():
            state.aggregator.reset_session()
            state.ticks_since_eval = 0
            state.last_candle_count = 0
            state.entries_this_session = 0
        self._cooldown.reset_session()
        self._generator.reset_session()
        self._sentinel.reset_session()
        self._fills.reset_session()
        self._positions.reset_session()
        self._pnl.reset_session()
        self._risk.reset_vwap_session()
        self.stats = BrainStats()
